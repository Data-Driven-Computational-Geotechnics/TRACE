# model/01-trace — the Trace model

Reusable, experiment-agnostic implementation of **Trace** (a contact-memory graph
network for granular dynamics). Importable as the `tracegnn` package.

```
tracegnn/
├── model3d.py      # class Trace — dimension-parametrized (dim, default 3) + Trace_NoMemory.
│                   #   This file is the engine for BOTH 2D and 3D.
├── model2d.py      # Trace2D / Trace2D_NoMemory — thin wrappers fixing dim=2 (no duplicated logic)
├── layers.py       # MLP, RMSNorm, Normalizer, edge-memory variants
│                   #   (EdgeMemoryState/gru, EdgeSpatioTemporalMemory/st, ...),
│                   #   MemoryEdgeBlock (processor), ContactForceDecoder (dim-parametrized)
├── data.py         # DEM data pipeline (SyntheticDEMData, DEMDataset)
├── distributed.py  # PhysicsNeMo-style multi-GPU harness (optional)
└── __init__.py     # exports: Trace, Trace_NoMemory, Trace2D, Trace2D_NoMemory, layers, data
```

Spatial dimension is a parameter: `Trace(dim=3)` (default) or `Trace(dim=2)`;
`Trace2D` is just `Trace(dim=2)`. The decoder, contact-force read-out, accel
normalizer, and box boundary all adapt to `dim`. Existing 3D checkpoints load
unchanged (default `dim=3`).

## Use

Experiment scripts add this directory to `sys.path` and `import tracegnn`:

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "01-trace"))
from tracegnn.model3d import Trace, Trace_NoMemory   # 3D
from tracegnn.model2d import Trace2D                 # 2D
# or: from tracegnn import Trace, Trace2D
```

(Or `pip install -e model/01-trace/` once a `pyproject.toml` is added, to drop the
path shim.)

## Baselines

The **GNS baseline is not separate code** — it is the memoryless variant
`Trace_NoMemory`, selected at the experiment level via `--memory none`. A standalone
`model/02-gns/` could be added later as a thin wrapper if an explicit baseline
folder is preferred.
