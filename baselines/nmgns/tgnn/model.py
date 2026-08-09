"""TGNNS — Temporal Graph Neural Network-based Simulator.

Faithful re-implementation of

    Shiwei Zhao, Hao Chen, Jidong Zhao,
    "A physical-information-flow-constrained temporal graph neural network-based
     simulator for granular materials", Comput. Methods Appl. Mech. Engrg.
     433 (2025) 117536.  doi:10.1016/j.cma.2024.117536.

This is the closest *domain* competitor to Trace and the foil for the project's
edge-memory-vs-node-memory comparison.

Key architecture (verified against the paper, Sections 2-3 and Table 1):

  * The input is a SEQUENCE of `n_frames` position snapshots. Each frame is turned
    into its OWN radius graph (dynamic graphs), processed as a TGNN "layer".
  * NODE-LEVEL temporal memory (NOT edge, NOT a GRU): per frame t a node historical
    state h_s^(t) is produced by one message-passing step (Eq.7); it is carried to
    the next frame and fused with that frame's freshly-encoded node embedding by a
    "node fusion" MLP gamma (Eq.8).  First frame has no history (Eq.9) -> separate
    encoders A (first) / B (subsequent).
  * PHYSICAL-INFORMATION-FLOW CONSTRAINT (the paper's headline novelty): between
    frames, ONLY the node state is carried forward; edge embeddings are DROPPED.
    Inter-graph information thus flows exclusively through nodes, mimicking
    Lagrangian particle methods (MPM) where state lives on material points.
  * After the last frame, a conventional GNN does `n_gnn` extra message-passing
    steps on the last graph; a decoder MLP outputs acceleration; semi-implicit
    Euler integrates.

Defaults follow the paper's baseline: 6 frames, 2 trailing GNN layers, latent 128,
2-hidden-layer MLPs, LeakyReLU(0.1), LayerNorm after outputs (except the decoder).
Dimension-parametrized (`dim=2` or `3`); same contact-radius graph and integrator
as Trace/GNS for a fair comparison.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def build_mlp(in_dim, out_dim, hidden=128, n_hidden=2, layernorm=True, slope=0.1):
    """Paper's MLP: `n_hidden` × (Linear + LeakyReLU(0.1)), final Linear, optional LayerNorm."""
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.LeakyReLU(slope)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    if layernorm:
        layers += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """Frozen per-channel z-score (fit once on the training set)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.eps = eps

    def set_stats(self, mean, std):
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_(std.clamp_min(self.eps).to(self.std))

    def normalize(self, x):   return (x - self.mean) / self.std
    def denormalize(self, x): return x * self.std + self.mean


class TGNNS(nn.Module):
    """Temporal Graph Neural Network-based Simulator (Zhao et al., CMAME 2025)."""

    def __init__(self, dim: int = 2, n_frames: int = 6, n_gnn: int = 2,
                 hidden_dim: int = 128, n_types: int = 3, skin_factor: float = 1.25):
        super().__init__()
        self.dim = dim
        self.n_frames = n_frames
        self.n_gnn = n_gnn
        self.hidden_dim = hidden_dim
        self.skin_factor = skin_factor
        nd = dim

        # --- four encoders (A = first frame, B = subsequent), Table 1 ---
        self.node_enc_A = build_mlp(2 * nd, hidden_dim)        # first frame: boundary proximity d_p (2·dim)
        self.node_enc_B = build_mlp(3 * nd, hidden_dim)        # subsequent: [velocity(dim), d_p(2·dim)]
        self.edge_enc_A = build_mlp(nd + 1, hidden_dim)        # first frame: displacement (unit dir + norm)
        self.edge_enc_B = build_mlp(2 * nd + 1, hidden_dim)    # subsequent: [rel_vel(dim), displacement(dim+1)]

        # --- node fusion gamma (Eq.8): fuse previous node historical state + current node embedding ---
        self.fusion = build_mlp(2 * hidden_dim, hidden_dim)

        # --- per-frame intra-graph message passing (g_theta, f_alpha); own params per frame ---
        self.frame_g = nn.ModuleList([build_mlp(3 * hidden_dim, hidden_dim) for _ in range(n_frames)])
        self.frame_f = nn.ModuleList([build_mlp(2 * hidden_dim, hidden_dim) for _ in range(n_frames)])

        # --- trailing conventional GNN (n_gnn message-passing layers on the last graph) ---
        self.gnn_g = nn.ModuleList([build_mlp(3 * hidden_dim, hidden_dim) for _ in range(n_gnn)])
        self.gnn_f = nn.ModuleList([build_mlp(2 * hidden_dim, hidden_dim) for _ in range(n_gnn)])

        self.decoder = build_mlp(hidden_dim, dim, hidden_dim, layernorm=False)   # -> normalized accel

        self.vel_norm = Normalizer(dim)
        self.accel_norm = Normalizer(dim)

    def set_accel_stats(self, mean, std): self.accel_norm.set_stats(mean, std)
    def set_vel_stats(self, mean, std):   self.vel_norm.set_stats(mean, std)

    # ---------------------------------------------------------------
    def _graph(self, pos, radius):
        """Radius graph on the contact threshold (skin·(r_i+r_j)); displacement edge feature."""
        N = pos.shape[0]
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)             # (N,N,dim)
        dist = diff.norm(dim=-1)
        thr = self.skin_factor * (radius.unsqueeze(0) + radius.unsqueeze(1))
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        idx = ((dist < thr) & (~eye)).nonzero(as_tuple=False)  # (E,2): send, recv
        send, recv = idx[:, 0], idx[:, 1]
        edge_index = torch.stack([send, recv], dim=0)
        rel = pos[send] - pos[recv]
        R_edge = (self.skin_factor * (radius[send] + radius[recv])).unsqueeze(-1)
        dnorm = rel.norm(dim=-1, keepdim=True)
        disp = torch.cat([rel / (dnorm + 1e-8), dnorm / R_edge], dim=-1)   # [unit dir (dim), |rel|/R (1)]
        return edge_index, disp

    def _boundary_prox(self, pos, radius, box):
        """Proximity to the 2·dim walls, in [0,1]; 0 (placeholder) beyond the influence radius."""
        lo, hi = box
        R = (self.skin_factor * 2.0 * radius).unsqueeze(-1)
        p_lo = (1.0 - (pos - lo) / R).clamp(0.0, 1.0)         # (N,dim) near low walls
        p_hi = (1.0 - (hi - pos) / R).clamp(0.0, 1.0)         # (N,dim) near high walls
        return torch.cat([p_lo, p_hi], dim=-1)               # (N, 2·dim)

    def _mp(self, h_v, h_e, edge_index, g, f):
        """One message-passing step (Eq.7-style): msg=g([h_w,h_v,h_e]), sum-aggregate, f([h_v,agg])."""
        send, recv = edge_index[0], edge_index[1]
        msg = g(torch.cat([h_v[send], h_v[recv], h_e], dim=-1))
        agg = scatter(msg, recv, dim=0, dim_size=h_v.shape[0], reduce="sum")
        return f(torch.cat([h_v, agg], dim=-1))

    # ---------------------------------------------------------------
    def forward(self, pos_seq, node_type, radius, dt=1.0, box=(0.0, 1.0)):
        """
        pos_seq:   (L, N, dim)  L = n_frames position snapshots, oldest -> newest
        node_type: (N,) long ;  radius: (N,)
        Returns 'accel' (normalized, for loss) and 'accel_phys' (for integration)
        for the LAST (newest) frame.
        """
        L = pos_seq.shape[0]
        h_s = None
        last_ei = last_he = None
        for i in range(L):
            pos = pos_seq[i]
            edge_index, disp = self._graph(pos, radius)
            d_p = self._boundary_prox(pos, radius, box)
            if i == 0:
                # first frame: encoder A, no history, no velocity (Eq.9)
                h_v = self.node_enc_A(d_p)
                h_e = self.edge_enc_A(disp)
            else:
                # subsequent frame: encoder B + node fusion with carried node state (Eq.8)
                v = self.vel_norm.normalize((pos - pos_seq[i - 1]) / dt)
                h_v0 = self.node_enc_B(torch.cat([v, d_p], dim=-1))
                h_v = self.fusion(torch.cat([h_s, h_v0], dim=-1))
                send, recv = edge_index[0], edge_index[1]
                rel_v = v[send] - v[recv]
                h_e = self.edge_enc_B(torch.cat([rel_v, disp], dim=-1))
            # intra-graph message passing -> node historical state (edges NOT carried forward)
            h_s = self._mp(h_v, h_e, edge_index, self.frame_g[i], self.frame_f[i])
            last_ei, last_he = edge_index, h_e

        # trailing conventional GNN on the last graph
        h_v = h_s
        for j in range(self.n_gnn):
            h_v = self._mp(h_v, last_he, last_ei, self.gnn_g[j], self.gnn_f[j])

        accel = self.decoder(h_v)                              # normalized accel (last frame)
        accel_phys = self.accel_norm.denormalize(accel)
        return {"accel": accel, "accel_phys": accel_phys, "n_edges": last_ei.shape[1]}

    @torch.no_grad()
    def rollout(self, pos_seq0, node_type, radius, n_steps, dt=1.0, box=(0.0, 1.0)):
        """Autoregressive rollout; maintains the L-frame position window. Boundary
        mirrors Trace (floor on axis 1 + side walls) for a fair comparison."""
        lo, hi = box
        pseq = pos_seq0.clone()                                # (L, N, dim)
        traj = []
        for _ in range(n_steps):
            a = self.forward(pseq, node_type, radius, dt=dt, box=box)["accel_phys"]
            v_new = (pseq[-1] - pseq[-2]) / dt + a * dt        # v_{t+1} = v_t + a*dt
            pos_new = pseq[-1] + v_new * dt                    # x_{t+1} = x_t + v_{t+1}*dt
            below = pos_new[:, 1] < lo + radius
            pos_new[below, 1] = lo + radius[below]
            for ax in [a_ for a_ in range(self.dim) if a_ != 1]:
                m = pos_new[:, ax] < lo + radius; pos_new[m, ax] = lo + radius[m]
                m = pos_new[:, ax] > hi - radius; pos_new[m, ax] = hi - radius[m]
            pseq = torch.cat([pseq[1:], pos_new.unsqueeze(0)], dim=0)   # shift window
            traj.append({"pos": pos_new.clone(), "vel": v_new.clone()})
        return traj


class TGNNS2D(TGNNS):
    def __init__(self, **kw): kw.setdefault("dim", 2); super().__init__(**kw)


class TGNNS3D(TGNNS):
    def __init__(self, **kw): kw.setdefault("dim", 3); super().__init__(**kw)
