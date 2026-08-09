"""2D Trace — thin wrappers over the dimension-parametrized model in model3d.py.

The Trace architecture (``model3d.py``) is parametrized by spatial dimension
``dim``. These classes simply fix ``dim=2`` for 2D particle/granular simulation;
the encoder, contact memory, message passing, force read-out, integration,
non-penetration projection and box boundary are ALL shared with the 3D model —
there is no duplicated logic. For 2D, positions/velocities/forces are 2-vectors
and the box boundary is floor (axis 1) + side walls on axis 0.
"""
from .model3d import Trace, Trace_NoMemory


class Trace2D(Trace):
    """2D Trace (``dim=2``). Identical architecture to the 3D model, restricted
    to a 2D spatial domain. All other hyper-parameters are forwarded unchanged."""

    def __init__(self, **kwargs):
        kwargs.setdefault("dim", 2)
        super().__init__(**kwargs)


class Trace2D_NoMemory(Trace_NoMemory):
    """2D memoryless control — the GNS baseline in 2D (``dim=2``)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("dim", 2)
        super().__init__(**kwargs)
