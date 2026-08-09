"""
008-Sand (2D) real-dataset loader for the Trace contact-memory GNN.

The dataset (DeepMind-GNS / LagrangeBench format) stores, per trajectory, only
``position`` (T, N, 2) and ``particle_type`` (N,) — all particles are sand
(type 6), and N varies trajectory-to-trajectory. The Trace model instead needs
``{pos, vel, radius, node_type}`` per trajectory (same dict the synthetic
``Collapse3DData`` produces), so this loader bridges the two:

  * velocity  — GNS finite-difference convention  v[t] = pos[t] - pos[t-1]
                (the recorded timestep is folded into the unit, i.e. dt = 1);
                v[0] = 0. The accel target is then a[t] = v[t+1] - v[t], the
                standard second-difference acceleration.
  * radius    — uniform, from the measured nearest-neighbour spacing
                (median ~0.0071 => r ~ 0.0036). Used both for building the
                contact graph (dist < skin_factor * 2r) and for the rollout
                non-penetration projection.
  * node_type — sand(6) is remapped to 0 ("normal particle"); there are no
                wall/boundary particles (the box is implicit).
  * box shift — positions are shifted by ``-box_lower`` so the domain becomes
                [0, box_size]^2, matching the Trace rollout box (floor at
                y = radius, side walls at [radius, box_size - radius]).

Outputs ``self.data``: a list of dicts ``{pos, vel, radius, node_type}`` with
pos/vel float32 (T, N, 2), radius float32 (N,), node_type int64 (N,).
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import h5py


class SandH5Data:
    def __init__(
        self,
        h5_path: str,
        n_samples: Optional[int] = None,
        radius: float = 0.0036,
        box_lower: float = 0.1,
        box_size: float = 0.8,
        start: int = 0,
        seed: int = 0,
    ):
        self.h5_path = str(h5_path)
        self.radius = float(radius)
        self.box_lower = float(box_lower)
        self.box_size = float(box_size)
        self.data = []

        with h5py.File(self.h5_path, "r") as h:
            keys = sorted(h.keys())
            keys = keys[start:]
            if n_samples is not None:
                keys = keys[:n_samples]
            for k in keys:
                g = h[k]
                pos = np.asarray(g["position"][:], dtype=np.float32)      # (T, N, 2)
                ptype = np.asarray(g["particle_type"][:], dtype=np.int64)  # (N,)
                N = pos.shape[1]

                # shift box [box_lower, box_lower+box_size] -> [0, box_size]
                pos = pos - self.box_lower

                # GNS finite-difference velocity (displacement; dt folded to 1)
                vel = np.zeros_like(pos)
                vel[1:] = pos[1:] - pos[:-1]

                radius_arr = np.full(N, self.radius, dtype=np.float32)
                # all particles are sand -> the model's "normal particle" type 0
                node_type = np.zeros(N, dtype=np.int64)
                _ = ptype  # kept for clarity; uniform sand, no walls

                self.data.append(dict(
                    pos=pos, vel=vel, radius=radius_arr, node_type=node_type,
                ))

    def __len__(self):
        return len(self.data)


def metadata_radius(meta_path: str, skin_factor: float):
    """Pick a uniform particle radius so the contact graph reproduces the
    dataset's ``default_connectivity_radius``:  skin_factor * 2r = R_conn."""
    meta = json.load(open(meta_path))
    R_conn = float(meta["default_connectivity_radius"])
    return R_conn / (2.0 * skin_factor), R_conn
