"""
Core GNN layers for Contact-Memory Graph Networks (Trace).

Key pieces:
1. EDGE MEMORY (`EdgeMemoryState`) — persistent GRU-style recurrent state per
   contact edge, born when a contact forms, updated ONCE per simulation step,
   discarded at contact death. This is the Trace core novelty.
2. MEMORY-AWARE PROCESSOR (`MemoryEdgeBlock`) — latent message passing that
   reads the persistent edge memory. (Linear-momentum conservation is NOT done
   here; it is realized in the model's antisymmetric force read-out, see model.py.)
3. CONTACT FORCE DECODER (`ContactForceDecoder`) — per-contact normal + Coulomb
   clamped tangential force from the edge memory; these forces are applied
   pairwise (+f to i, -f to j) in model.py to conserve linear momentum exactly.
4. `Normalizer` — frozen mean/std (registered buffers) for the acceleration
   target so the loss is well-conditioned (the missing piece that made the old
   loss plateau on the gravity-dominated, un-normalized target).

NOTE: `EdgeMemoryState` is an nn.Module so its GRU/birth params register in
`model.parameters()` and move with `.to(device)` (previously a plain class — its
params were never trained nor moved to GPU).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax, scatter
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════
# Utility layers
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Two-layer MLP with inner LayerNorm + SiLU.

    `final_act=False` gives a LINEAR output head — required for regression
    outputs that must span negative values (e.g. downward acceleration). The
    old code always ended in SiLU, which clamps outputs to >= -0.28 and made it
    impossible to predict gravity.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128,
                 final_act: bool = True):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ]
        if final_act:
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (lighter than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class Normalizer(nn.Module):
    """Frozen feature/target normalizer with mean/std stored as buffers.

    Stats are fit ONCE from the training set (before training) and frozen — they
    save/load in state_dict and move with `.to(device)` and broadcast under DDP.
    """

    def __init__(self, dim: int, eps: float = 1e-6, center: bool = True):
        super().__init__()
        self.eps = eps
        self.center = center                  # False => scale-only: mean stays 0
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        """x: (..., dim). Per-channel mean/std.

        Degenerate (near-zero-variance) channels — e.g. the z-channel of the
        2D-padded-to-3D data — get std=1.0, NOT a tiny eps. A tiny eps would
        avoid NaN but blow up that channel's normalized error and dominate the
        loss/gradients; std=1.0 leaves such channels at their natural (~0) scale.

        ``center=False`` (scale-only) leaves mean at 0 and measures std ABOUT
        ZERO (sqrt(E[x²])), so that ``normalize(-x) == -normalize(x)`` holds
        exactly — required for the antisymmetric reverse-edge features.
        """
        flat = x.reshape(-1, x.shape[-1]).float()
        if self.center:
            self.mean.copy_(flat.mean(dim=0))
            raw_std = flat.std(dim=0)
        else:
            self.mean.zero_()
            raw_std = flat.pow(2).mean(dim=0).sqrt()
        safe_std = torch.where(raw_std < self.eps, torch.ones_like(raw_std), raw_std)
        self.std.copy_(safe_std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


# ═══════════════════════════════════════════════════════════════════
# Edge Memory — the core Trace innovation (now an nn.Module)
# ═══════════════════════════════════════════════════════════════════

class EdgeMemoryState(nn.Module):
    """
    Persistent per-contact-edge memory across timesteps.

    Each active contact edge (i,j) carries a learned vector `m_ij`:
    - BIRTH: when contact forms, initialized from relative state + learned bias
    - UPDATE: GRU-style gated update ONCE per simulation step
    - DEATH: when contact breaks, state is discarded

    Difference from TGNNS (discards edge state between steps) and from
    Dynami-CAL (only within-step edge messages): here `m_ij` persists across
    simulation steps and is the input to the contact-force read-out.
    """

    def __init__(self, memory_dim: int = 16, hidden_dim: int = 128,
                 edge_in_dim: int = 7):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.carry_dim = memory_dim          # width of the carried-across-steps state
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=memory_dim)
        # Birth init: relative state at contact formation -> initial memory.
        # final_act=False so the init can span the full memory range.
        self.birth_init = MLP(edge_in_dim, memory_dim, hidden_dim, final_act=False)

    def initialize(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """edge_attr: (E, edge_in_dim) -> m_init: (E, carry_dim)."""
        return self.birth_init(edge_attr)

    def current(self, carry: torch.Tensor) -> torch.Tensor:
        """Extract the (E, memory_dim) memory used by processor/decoder from the
        carried state. For the plain GRU the carry IS the memory."""
        return carry

    def update(self, m_old: torch.Tensor, edge_features: torch.Tensor,
               edge_index: torch.Tensor = None) -> torch.Tensor:
        """GRU-style gated update -> new carry (E, memory_dim).
        `edge_index` accepted for interface parity (ignored — purely per-edge)."""
        return self.gru(edge_features, m_old)


class EdgeSpatioTemporalMemory(nn.Module):
    """时空记忆边 — SPATIO-TEMPORAL per-contact edge memory (the method to validate).

    TEMPORAL: a GRU carries each contact's memory across simulation steps (as in
    EdgeMemoryState).  SPATIAL: before the GRU update, each contact edge gathers
    information from the OTHER contact edges that SHARE one of its two particles
    (the line graph) via an efficient O(E) node-mediated attention — every edge
    contributes to its two endpoint particles; per particle we attention-pool the
    incident edges; each edge reads back the pooled context of its two endpoints.
    The attended spatial context is concatenated with the edge latent and fed to
    the temporal GRU.  Physically: a contact "sees" the other contacts on the same
    grain (multi-contact coupling / force chains) AND remembers its own history.
    """

    def __init__(self, memory_dim: int = 16, hidden_dim: int = 128, edge_in_dim: int = 7,
                 heads: int = 1):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.carry_dim = memory_dim
        self.heads = heads
        self.hd = hidden_dim // heads                 # per-head dim
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(heads, self.hd) * 0.1)  # per-head query
        self.scale = self.hd ** -0.5
        self.gru = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=memory_dim)
        self.birth_init = MLP(edge_in_dim, memory_dim, hidden_dim, final_act=False)

    def initialize(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.birth_init(edge_attr)

    def current(self, carry: torch.Tensor) -> torch.Tensor:
        return carry

    def _spatial(self, edge_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        E = edge_h.shape[0]
        row, col = edge_index[0], edge_index[1]
        N = int(torch.max(edge_index).item()) + 1 if E > 0 else 1
        H, hd = self.heads, self.hd
        k = self.k_proj(edge_h).view(E, H, hd); v = self.v_proj(edge_h).view(E, H, hd)
        node = torch.cat([row, col], dim=0)
        kk = torch.cat([k, k], dim=0); vv = torch.cat([v, v], dim=0)        # (2E,H,hd)
        score = (kk * self.query).sum(-1) * self.scale                     # (2E,H)
        attn = pyg_softmax(score, node, num_nodes=N)                       # per-node per-head
        node_ctx = scatter(attn.unsqueeze(-1) * vv, node, dim=0, dim_size=N, reduce="sum")
        node_ctx = node_ctx.reshape(N, H * hd)
        return 0.5 * (node_ctx[row] + node_ctx[col])

    def update(self, m_old: torch.Tensor, edge_features: torch.Tensor,
               edge_index: torch.Tensor = None) -> torch.Tensor:
        if edge_index is None or edge_features.shape[0] == 0:
            ctx = torch.zeros_like(edge_features)
        else:
            ctx = self._spatial(edge_features, edge_index)
        return self.gru(torch.cat([edge_features, ctx], dim=-1), m_old)


class EdgeSpatialOnly(nn.Module):
    """纯空间(消融变体, memory_type='spatial'): 每步用线图注意力算 s_ij^t 投影成记忆喂解码器,
    但【不跨步携带任何时间状态】(忽略 m_old, 每步现算)。用于 2x2 消融, 隔离"空间聚合"
    独立于时间记忆的贡献。注意: 无时间持久性, 严格讲是一次线图消息传递, 不是"记忆"。"""

    def __init__(self, memory_dim: int = 16, hidden_dim: int = 128, edge_in_dim: int = 7):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.carry_dim = memory_dim
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.scale = hidden_dim ** -0.5
        self.out = nn.Linear(hidden_dim, memory_dim)                 # s_ij^t -> memory_dim
        self.birth_init = MLP(edge_in_dim, memory_dim, hidden_dim, final_act=False)  # 接口占位

    def initialize(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.birth_init(edge_attr)

    def current(self, carry: torch.Tensor) -> torch.Tensor:
        return carry

    def _spatial(self, edge_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        E = edge_h.shape[0]
        row, col = edge_index[0], edge_index[1]
        N = int(torch.max(edge_index).item()) + 1 if E > 0 else 1
        k = self.k_proj(edge_h); v = self.v_proj(edge_h)            # (E, hidden)
        node = torch.cat([row, col], dim=0)
        kk = torch.cat([k, k], dim=0); vv = torch.cat([v, v], dim=0)
        score = (kk * self.query).sum(-1) * self.scale             # (2E,)
        attn = pyg_softmax(score, node, num_nodes=N)
        node_ctx = scatter(attn.unsqueeze(-1) * vv, node, dim=0, dim_size=N, reduce="sum")
        return 0.5 * (node_ctx[row] + node_ctx[col])               # (E, hidden)

    def update(self, m_old: torch.Tensor, edge_features: torch.Tensor,
               edge_index: torch.Tensor = None) -> torch.Tensor:
        # 忽略 m_old: 无时间记忆, 每步现算空间上下文
        if edge_index is None or edge_features.shape[0] == 0:
            ctx = torch.zeros_like(edge_features)
        else:
            ctx = self._spatial(edge_features, edge_index)
        return self.out(ctx)                                        # (E, memory_dim)


class EdgeSpatioTemporalAttn(nn.Module):
    """st2 — genuine SPATIO-TEMPORAL ATTENTION edge memory (the 时空注意力 test).

    Carries a WINDOW of the last `window` memory frames per contact edge.
    Each step: (a) TEMPORAL attention — the latest frame (query) attends over the
    window of past frames (keys/values); (b) SPATIAL attention — line-graph
    attention over neighbour edges sharing a particle (as in EdgeSpatioTemporalMemory);
    (c) fuse [edge_latent ; spatial_ctx ; temporal_ctx] through a GRU on the latest
    frame to produce the new frame; (d) shift the window. Unlike `gru`/`st` whose
    temporal mechanism is a single-step GRU, here time is handled by ATTENTION over
    a bounded window (the bounded window also resists the long-rollout drift that a
    pure recurrence accumulates).
    """

    def __init__(self, memory_dim: int = 16, hidden_dim: int = 128, edge_in_dim: int = 7,
                 window: int = 4):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.W = window
        self.carry_dim = memory_dim * window           # window of W frames, flattened
        # spatial (line-graph) attention
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.s_query = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.s_scale = hidden_dim ** -0.5
        # temporal attention over the window
        self.t_q = nn.Linear(memory_dim, memory_dim)
        self.t_k = nn.Linear(memory_dim, memory_dim)
        self.t_v = nn.Linear(memory_dim, memory_dim)
        self.t_scale = memory_dim ** -0.5
        # fuse [edge_h ; spatial_ctx ; temporal_ctx] -> new frame
        self.gru = nn.GRUCell(input_size=hidden_dim * 2 + memory_dim, hidden_size=memory_dim)
        self.birth_init = MLP(edge_in_dim, memory_dim, hidden_dim, final_act=False)

    def initialize(self, edge_attr: torch.Tensor) -> torch.Tensor:
        m0 = self.birth_init(edge_attr)                # (E, mem)
        return m0.repeat(1, self.W)                    # (E, W*mem): tile birth across window

    def current(self, carry: torch.Tensor) -> torch.Tensor:
        E = carry.shape[0]
        if E == 0:
            return carry.new_zeros((0, self.memory_dim))
        return carry.view(E, self.W, self.memory_dim)[:, -1]   # latest frame

    def _spatial(self, edge_h, edge_index):
        E = edge_h.shape[0]
        row, col = edge_index[0], edge_index[1]
        N = int(torch.max(edge_index).item()) + 1 if E > 0 else 1
        k = self.k_proj(edge_h); v = self.v_proj(edge_h)
        node = torch.cat([row, col], dim=0)
        kk = torch.cat([k, k], dim=0); vv = torch.cat([v, v], dim=0)
        score = (kk * self.s_query).sum(-1) * self.s_scale
        attn = pyg_softmax(score, node, num_nodes=N)
        node_ctx = scatter(attn.unsqueeze(-1) * vv, node, dim=0, dim_size=N, reduce="sum")
        return 0.5 * (node_ctx[row] + node_ctx[col])

    def update(self, carried: torch.Tensor, edge_features: torch.Tensor,
               edge_index: torch.Tensor = None) -> torch.Tensor:
        E = edge_features.shape[0]
        if E == 0:
            return carried
        win = carried.view(E, self.W, self.memory_dim)         # (E, W, mem)
        latest = win[:, -1]                                    # (E, mem)
        # temporal attention: latest queries the window
        q = self.t_q(latest).unsqueeze(1)                      # (E,1,mem)
        kt = self.t_k(win); vt = self.t_v(win)                 # (E,W,mem)
        a_t = torch.softmax((q * kt).sum(-1) * self.t_scale, dim=-1)  # (E,W)
        t_ctx = (a_t.unsqueeze(-1) * vt).sum(1)                # (E,mem)
        # spatial attention
        s_ctx = self._spatial(edge_features, edge_index) if edge_index is not None \
            else torch.zeros_like(edge_features)
        new = self.gru(torch.cat([edge_features, s_ctx, t_ctx], dim=-1), latest)  # (E,mem)
        new_win = torch.cat([win[:, 1:], new.unsqueeze(1)], dim=1)                # shift
        return new_win.reshape(E, self.W * self.memory_dim)


# ═══════════════════════════════════════════════════════════════════
# Memory-aware processor block (latent message passing)
# ═══════════════════════════════════════════════════════════════════

class MemoryEdgeBlock(MessagePassing):
    """
    Latent edge->node message passing that reads the persistent edge memory.

    This produces updated node latents; it does NOT itself conserve momentum.
    Linear-momentum conservation is enforced separately by the antisymmetric
    force read-out in Trace.forward (Newton's third law: +f to i, -f to j).
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, memory_dim: int = 16):
        super().__init__(aggr="sum")
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.memory_dim = memory_dim

        total_edge_dim = edge_feat_dim + memory_dim + node_feat_dim * 2
        self.edge_mlp = MLP(total_edge_dim, node_feat_dim)
        self.node_mlp = MLP(node_feat_dim * 2, node_feat_dim)  # residual update
        self.node_norm = RMSNorm(node_feat_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_memory: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.shape[1] == 0:
            # No contacts: identity node update, empty messages.
            return x, x.new_zeros((0, self.node_feat_dim))
        edge_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                  edge_memory=edge_memory)
        x_new = self.node_mlp(torch.cat([x, edge_msg], dim=-1))
        x_new = x + self.node_norm(x_new)  # residual
        return x_new, edge_msg

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: torch.Tensor, edge_memory: torch.Tensor) -> torch.Tensor:
        return self.edge_mlp(torch.cat([edge_attr, edge_memory, x_i, x_j], dim=-1))


# Backwards-compatible alias (older imports / docs use this name).
MomentumConservingEdgeBlock = MemoryEdgeBlock


# ═══════════════════════════════════════════════════════════════════
# Contact Force Decoder
# ═══════════════════════════════════════════════════════════════════

class ContactForceDecoder(nn.Module):
    """
    Decode per-contact (normal magnitude, tangential vector, friction) from the
    edge memory + edge latent.

    The memory `m_ij` is the learned analogue of the Mindlin tangential spring.
    F_n >= 0 (softplus). F_t is clamped to the learned Coulomb cone |F_t| <= mu*F_n.
    These are turned into an antisymmetric pairwise force in Trace.forward.

    For the EMERGENT memory version (Trace headline), per-contact forces are NOT
    supervised — the memory must learn friction history from macro observables.
    """

    def __init__(self, memory_dim: int = 16, edge_dim: int = 128, hidden_dim: int = 128, dim: int = 3):
        super().__init__()
        self.force_mlp = MLP(memory_dim + edge_dim, hidden_dim, hidden_dim)
        self.normal_head = nn.Linear(hidden_dim, 1)
        self.tangent_head = nn.Linear(hidden_dim, dim)
        self.friction_head = nn.Linear(hidden_dim, 1)

    def forward(self, edge_memory: torch.Tensor, edge_attr: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns F_n:(E,1)>=0, F_t:(E,3) Coulomb-clamped, mu:(E,1) in [0.1,1.0]."""
        h = self.force_mlp(torch.cat([edge_memory, edge_attr], dim=-1))
        F_n = torch.nn.functional.softplus(self.normal_head(h))
        F_t_raw = self.tangent_head(h)
        mu = torch.sigmoid(self.friction_head(h)) * 0.9 + 0.1

        F_t_norm = torch.norm(F_t_raw, dim=-1, keepdim=True)
        clamp = torch.clamp(mu * F_n / (F_t_norm + 1e-8), max=1.0)
        F_t = F_t_raw * clamp
        return F_n, F_t, mu
