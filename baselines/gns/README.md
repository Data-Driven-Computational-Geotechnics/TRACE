# model/02-gns — GNS baseline

Faithful re-implementation of the standard **Graph Network-based Simulator (GNS)**:

> Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec, Battaglia.
> *Learning to Simulate Complex Physics with Graph Networks.* ICML 2020.
> arXiv:2002.09405 (DeepMind `learning_to_simulate`).

This is the **node-level memory baseline** for Trace: where Trace carries memory on
contact *edges*, GNS carries temporal memory in *node* features (a short velocity
history). The two thus form the key "edge-memory vs node-memory" comparison.

```
gns/
├── model.py   # GNS (+ GNS2D/GNS3D), GNBlock (full GN processor block), Normalizer
└── __init__.py
```

## Architecture (as in the paper)

- **Encoder** — node features = velocity history (`n_history`=5 frames) + clipped
  distances to the box walls + particle-type embedding (+ radius for granular);
  edge features = relative displacement + its norm. Two MLP encoders → 128-d latents.
- **Processor** — `num_layers`=10 full Graph-Network blocks; each updates BOTH edge
  and node latents with residual connections (ReLU MLPs + LayerNorm).
- **Decoder** — a per-node MLP reads out the (normalized) acceleration. No edge
  memory, no force decomposition, no momentum conservation.
- **Integration** — semi-implicit Euler (`rollout`), maintaining the velocity-history
  window; boundary handling mirrors Trace for a fair comparison.

Dimension-parametrized: `GNS(dim=2|3)`, or `GNS2D` / `GNS3D`.

## Differences vs Trace (the comparison)

| | Trace (`model/01-trace`) | GNS (here) |
|---|---|---|
| Temporal memory | per-**contact edge** GRU | **node** velocity history |
| Processor | node-only update; edges/memory frozen | full GN block (edge + node) |
| Force read-out | structured (normal + Coulomb tangential), momentum-conserving | plain node → acceleration |

## Status / how to use

The model is verified by a forward + rollout smoke test (2D and 3D). To train and
compare it against Trace it needs **velocity-history input** (`vel_seq`, shape
`(N, n_history, dim)`); the current `experiments/01-3d-collapse` data pipeline feeds
single-frame velocity, so a small adapter (stack the last `n_history` velocities) is
the remaining integration step. This GNS baseline directly enables the
edge-memory-vs-node-memory experiment the paper needs.
