"""GNS — Graph Network-based Simulator (standard baseline).

Faithful re-implementation of the canonical simulator of

    Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec, Battaglia,
    "Learning to Simulate Complex Physics with Graph Networks", ICML 2020.
    arXiv:2002.09405  (DeepMind `learning_to_simulate`).

This is the standard baseline against which Trace (contact-memory) is compared.
Key architectural choices that follow the paper (and differ from Trace):

  * Node features carry a SHORT VELOCITY HISTORY (the last `n_history` velocities)
    — this is GNS's node-level temporal memory, the counterpart to Trace's
    per-contact edge memory.
  * The processor is a stack of full Graph-Network blocks: every step updates
    BOTH edge and node latents (with residual connections).
  * The decoder is a plain per-node MLP that reads out acceleration. There is NO
    edge memory, NO force decomposition, NO momentum-conservation / Coulomb cone.

For a fair comparison with Trace in this project, the graph is built with the
same contact radius (skin_factor·(r_i+r_j)) and the state is advanced with the
same semi-implicit Euler integrator. The model is dimension-parametrized
(`dim=2` or `dim=3`).
"""
import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def build_mlp(in_dim: int, out_dim: int, hidden: int = 128,
              n_hidden_layers: int = 2, layernorm: bool = True) -> nn.Sequential:
    """GNS-style MLP: `n_hidden_layers` × (Linear+ReLU), final Linear, optional LayerNorm."""
    layers, d = [], in_dim
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    if layernorm:
        layers += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """Frozen per-channel z-score. Stats are fit once on the training set."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.eps = eps

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_(std.clamp_min(self.eps).to(self.std))

    def normalize(self, x):   return (x - self.mean) / self.std
    def denormalize(self, x): return x * self.std + self.mean


class GNBlock(nn.Module):
    """One Graph-Network message-passing step (the GNS 'interaction network' block).

    Edge update:  e_ij <- e_ij + phi_e([e_ij, v_send, v_recv])
    Node update:  v_i  <- v_i  + phi_v([v_i, sum_j e_ji])
    Both use residual connections, as in the paper.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.edge_mlp = build_mlp(3 * hidden, hidden, hidden)   # [e_ij, v_i, v_j]
        self.node_mlp = build_mlp(2 * hidden, hidden, hidden)   # [v_i, aggregated edges]

    def forward(self, node_h, edge_h, edge_index):
        if edge_index.shape[1] == 0:
            # no edges: node update with zero aggregate
            agg = node_h.new_zeros((node_h.shape[0], node_h.shape[1]))
            node_h = node_h + self.node_mlp(torch.cat([node_h, agg], dim=-1))
            return node_h, edge_h
        send, recv = edge_index[0], edge_index[1]
        e_upd = self.edge_mlp(torch.cat([edge_h, node_h[send], node_h[recv]], dim=-1))
        edge_h = edge_h + e_upd                                                  # residual edge
        agg = scatter(edge_h, recv, dim=0, dim_size=node_h.shape[0], reduce="sum")
        node_h = node_h + self.node_mlp(torch.cat([node_h, agg], dim=-1))        # residual node
        return node_h, edge_h


class GNS(nn.Module):
    """Graph Network-based Simulator (Sanchez-Gonzalez et al., ICML 2020)."""

    def __init__(self, dim: int = 3, n_history: int = 5, n_types: int = 3,
                 hidden_dim: int = 128, num_layers: int = 10,
                 skin_factor: float = 1.25, type_emb_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.n_history = n_history
        self.skin_factor = skin_factor
        self.hidden_dim = hidden_dim

        self.type_emb = nn.Embedding(n_types, type_emb_dim)
        # node input = velocity history (n_history·dim) + clipped wall distances (2·dim)
        #            + particle-type embedding + radius(1)
        node_in = n_history * dim + 2 * dim + type_emb_dim + 1
        edge_in = dim + 1                                       # rel_pos(dim) + |rel_pos|(1)

        self.node_encoder = build_mlp(node_in, hidden_dim, hidden_dim)
        self.edge_encoder = build_mlp(edge_in, hidden_dim, hidden_dim)
        self.processor = nn.ModuleList([GNBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = build_mlp(hidden_dim, dim, hidden_dim, layernorm=False)   # -> normalized accel

        self.vel_norm = Normalizer(n_history * dim)
        self.accel_norm = Normalizer(dim)

    # --- fit normalizer stats once before training ---
    def set_accel_stats(self, mean, std): self.accel_norm.set_stats(mean, std)
    def set_vel_stats(self, mean, std):   self.vel_norm.set_stats(mean, std)

    def _build_graph(self, pos, radius):
        """Radius graph on the same contact threshold as Trace (skin·(r_i+r_j))."""
        N = pos.shape[0]
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)              # (N,N,dim): pos[b]-pos[a]
        dist = diff.norm(dim=-1)
        thr = self.skin_factor * (radius.unsqueeze(0) + radius.unsqueeze(1))
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        mask = (dist < thr) & (~eye)
        idx = mask.nonzero(as_tuple=False)                     # (E,2): (send a, recv b)
        send, recv = idx[:, 0], idx[:, 1]
        edge_index = torch.stack([send, recv], dim=0)          # (2,E)
        rel = pos[send] - pos[recv]                            # displacement
        R_edge = (self.skin_factor * (radius[send] + radius[recv])).unsqueeze(-1)  # (E,1) connectivity radius
        rel = rel / R_edge                                     # normalize by connectivity radius (canonical GNS)
        edge_feat = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=-1)   # (E, dim+1): (rel/R, |rel|/R)
        return edge_index, edge_feat

    def forward(self, pos, vel_seq, node_type, radius, box=(0.0, 1.0)):
        """
        pos:       (N, dim)            current positions
        vel_seq:   (N, n_history, dim) last `n_history` velocities (oldest..newest)
        node_type: (N,)  long          particle type ids
        radius:    (N,)                particle radii
        box:       (lo, hi)            axis-aligned box bounds (for wall features)
        Returns dict with 'accel' (normalized, for loss) and 'accel_phys' (for integration).
        """
        N = pos.shape[0]
        lo, hi = box

        # --- node features: normalized velocity history + wall distances + type + radius ---
        vflat = self.vel_norm.normalize(vel_seq.reshape(N, -1))             # (N, n_history*dim)
        R = (self.skin_factor * 2.0 * radius).unsqueeze(-1)                # (N,1) per-particle connectivity radius
        d_lo = ((pos - lo) / R).clamp(-1.0, 1.0)                           # signed dist to low walls, in [-1,1] (canonical)
        d_hi = ((hi - pos) / R).clamp(-1.0, 1.0)                           # signed dist to high walls, in [-1,1]
        temb = self.type_emb(node_type)                                    # (N, type_emb_dim)
        node_feat = torch.cat([vflat, d_lo, d_hi, temb, radius.unsqueeze(-1)], dim=-1)

        node_h = self.node_encoder(node_feat)
        edge_index, edge_feat = self._build_graph(pos, radius)
        edge_h = (self.edge_encoder(edge_feat) if edge_index.shape[1] > 0
                  else node_h.new_zeros((0, self.hidden_dim)))

        for blk in self.processor:
            node_h, edge_h = blk(node_h, edge_h, edge_index)

        accel = self.decoder(node_h)                                       # normalized accel
        accel_phys = self.accel_norm.denormalize(accel)                    # physical accel
        return {"accel": accel, "accel_phys": accel_phys,
                "n_edges": edge_index.shape[1]}

    @torch.no_grad()
    def rollout(self, pos0, vel_seq0, node_type, radius, n_steps, dt=1.0, box=(0.0, 1.0)):
        """Autoregressive rollout with semi-implicit Euler, maintaining the velocity
        history window. Boundary mirrors the Trace convention (floor on axis 1 +
        side walls on the other axes)."""
        lo, hi = box
        pos = pos0.clone()
        vseq = vel_seq0.clone()                                            # (N, n_history, dim)
        traj = []
        for _ in range(n_steps):
            a = self.forward(pos, vseq, node_type, radius, box=box)["accel_phys"]
            v_new = vseq[:, -1] + a * dt                                   # v_{t+1} = v_t + a*dt
            pos = pos + v_new * dt                                         # x_{t+1} = x_t + v_{t+1}*dt
            # floor (axis 1, down only)
            below = pos[:, 1] < lo + radius
            pos[below, 1] = lo + radius[below]
            v_new[below, 1] = v_new[below, 1].clamp(min=0)
            # side walls (other axes, both sides)
            for ax in [a_ for a_ in range(self.dim) if a_ != 1]:
                m = pos[:, ax] < lo + radius
                pos[m, ax] = lo + radius[m]; v_new[m, ax] = v_new[m, ax].clamp(min=0)
                m = pos[:, ax] > hi - radius
                pos[m, ax] = hi - radius[m]; v_new[m, ax] = v_new[m, ax].clamp(max=0)
            vseq = torch.cat([vseq[:, 1:], v_new.unsqueeze(1)], dim=1)     # shift history window
            traj.append({"pos": pos.clone(), "vel": v_new.clone()})
        return traj


class GNS2D(GNS):
    """2D GNS (dim=2)."""
    def __init__(self, **kwargs):
        kwargs.setdefault("dim", 2)
        super().__init__(**kwargs)


class GNS3D(GNS):
    """3D GNS (dim=3)."""
    def __init__(self, **kwargs):
        kwargs.setdefault("dim", 3)
        super().__init__(**kwargs)
