# model/03-tgnn — TGNNS baseline

Faithful re-implementation of the **Temporal Graph Neural Network-based Simulator (TGNNS)**:

> Shiwei Zhao, Hao Chen, Jidong Zhao.
> *A physical-information-flow-constrained temporal graph neural network-based
> simulator for granular materials.* Comput. Methods Appl. Mech. Engrg. **433**
> (2025) 117536. doi:10.1016/j.cma.2024.117536.
> (Author PDF: `1-s2.0-S0045782524007904-main.pdf`, included in this folder.)

This is the **closest domain competitor to Trace** and the foil for the project's
**edge-memory vs node-memory** comparison.

```
tgnn/
├── model.py   # TGNNS (+ TGNNS2D/3D), Normalizer, build_mlp
└── __init__.py
```

## Architecture (as in the paper, Sections 2–3, Table 1)

- **Input** = a sequence of `n_frames` position snapshots (baseline 6). Each frame
  becomes its own radius graph (dynamic graphs over time).
- **Temporal memory = NODE-level, not edge, not GRU.** Per frame, a node historical
  state `h_s` is produced by one message-passing step (Eq. 7) and carried to the next
  frame, fused with that frame's freshly-encoded node embedding by a node-fusion MLP
  (Eq. 8). The first frame has no history (Eq. 9) → separate encoders A (first) / B
  (subsequent): 4 encoders total.
- **Physical-information-flow constraint (headline novelty)** — between frames, ONLY
  the node state is passed forward; **edge embeddings are dropped**. Inter-graph
  information flows exclusively through nodes, mimicking Lagrangian particle methods
  (MPM). It is an *architectural* inductive bias, not a PINN-style loss penalty.
- **Trailing GNN** — after the last frame, `n_gnn` (baseline 2) conventional
  message-passing steps on the last graph.
- **Decoder** → acceleration; semi-implicit Euler integration.
- MLPs: 2 hidden layers, latent 128, **LeakyReLU(0.1)**, LayerNorm after outputs
  (except the decoder). Dimension-parametrized (`dim=2|3`).

## The three-way comparison this enables

| | Trace (`01-trace`) | GNS (`02-gns`) | TGNNS (`03-tgnn`) |
|---|---|---|---|
| Temporal memory | per-**contact edge** GRU | **node** velocity history (stacked) | **node** historical state across a frame *sequence* |
| Inter-step info flow | on edges (memory) | none (single graph) | **nodes only** (edges dropped = physical-flow constraint) |
| Graph | dynamic, per step | single (history in node feat) | dynamic, one per frame |

Notably, TGNNS **deliberately rejects edge memory** (its own ablation finds node-only
carry suffices) — making it the ideal foil for Trace's contact-edge-memory claim.

## Status / how to use

Verified by a forward + rollout smoke test (2D and 3D). Like GNS, it needs a
**position sequence** input (`pos_seq`, shape `(n_frames, N, dim)`); the current
`experiments/01-3d-collapse` pipeline feeds single frames, so a small adapter
(stack the last `n_frames` positions) is the remaining integration step. The
included paper PDF is the authoritative reference.
