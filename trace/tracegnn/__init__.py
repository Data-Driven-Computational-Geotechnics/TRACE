# Trace: Contact-Memory Graph Networks
# A GNN-DEM simulator with persistent per-contact-edge recurrent memory.
#
# This package is designed to be compatible with NVIDIA PhysicsNeMo's
# distributed training infrastructure. The model itself uses PyTorch
# Geometric (PyG) for graph operations.
#
# Reference papers:
#   - Dynami-CAL GraphNet (arXiv:2501.07373) — momentum conservation
#   - GNS (arXiv:2002.09405) — base encode-process-decode architecture

from .model3d import Trace, Trace_NoMemory          # 3D (dim=3); also dimension-parametrized engine
from .model2d import Trace2D, Trace2D_NoMemory       # 2D (dim=2) thin wrappers
from .layers import EdgeMemoryState, MomentumConservingEdgeBlock, ContactForceDecoder
from .data import DEMDataset, SyntheticDEMData, collate_graphs

__all__ = [
    "Trace",
    "Trace_NoMemory",
    "Trace2D",
    "Trace2D_NoMemory",
    "EdgeMemoryState",
    "MomentumConservingEdgeBlock",
    "ContactForceDecoder",
    "DEMDataset",
    "SyntheticDEMData",
    "collate_graphs",
]
