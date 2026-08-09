"""
Data pipeline for Trace.

Two modes:
1. SyntheticDEMData — synthetic 2D granular column-collapse (spring-dashpot +
   Coulomb friction) for quick testing. Contact forces are VECTORIZED (numpy
   broadcasting) so sub-stepping is feasible. Physics runs at a small `dt_phys`
   for stability but is RECORDED every `substeps` sub-steps, so each recorded
   transition (`dt` = dt_phys * substeps) shows visible motion. Generated data
   is cached to disk so DDP ranks don't each regenerate it.
2. DEMDataset — loads real DEM data from HDF5 files (LIGGGHTS/Yade output).

Designed to be compatible with PhysicsNeMo's datapipes + PyTorch DataLoader.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from multiprocessing import Pool
import h5py


# ═══════════════════════════════════════════════════════════════════
# Module-level trajectory generator (top-level so it is picklable for
# multiprocessing — independent trajectories generate in parallel).
# ═══════════════════════════════════════════════════════════════════

def _gen_traj(params):
    """params = (seed, N, T, L, dt, substeps, k_n, gamma_n) -> (pos3, vel3, radius, node_type).

    Proper granular COLUMN COLLAPSE: a tall narrow column rests ON the floor and
    is released; gravity makes it slump and spread (no high drop). Soft-sphere
    contacts are STIFF enough that inter-particle overlap stays small (a few % of
    the radius), unlike the old k_n=5000 + high-drop config which let particles
    pass through each other by up to a full diameter.
    """
    seed, N, T, L, dt, substeps, k_n, gamma_n = params
    rng = np.random.RandomState(seed)
    r = 0.012
    g = np.array([0.0, -9.81], dtype=np.float64)
    mu = 0.5
    dt_phys = dt / substeps

    # Tall column resting on the floor (bottom row at y=radius), centered.
    spacing = r * 2.02
    col_w = max(1, int(np.sqrt(N / 2.6)))          # narrow -> tall aspect ratio
    rows = int(np.ceil(N / col_w))
    pos = np.zeros((N, 2))
    for i in range(N):
        rr, cc = i // col_w, i % col_w
        pos[i, 0] = L * 0.5 + (cc - col_w / 2) * spacing + rng.uniform(-3e-4, 3e-4)
        pos[i, 1] = r + rr * spacing + rng.uniform(-3e-4, 3e-4)
    vel = np.zeros((N, 2))
    radius = np.ones(N) * r
    min_d = radius[:, None] + radius[None, :]

    pos_seq = np.zeros((T, N, 2)); vel_seq = np.zeros((T, N, 2))
    for t in range(T):
        for _ in range(substeps):
            d = pos[:, None, :] - pos[None, :, :]            # d[i,j] = pos_i - pos_j
            dist = np.sqrt((d ** 2).sum(-1))
            np.fill_diagonal(dist, np.inf)
            overlap = min_d - dist
            contact = overlap > 0
            n = d / (dist[..., None] + 1e-12)                # unit, points j -> i
            vr = vel[:, None, :] - vel[None, :, :]
            vrn = (vr * n).sum(-1)
            Fn = np.where(contact, np.clip(k_n * overlap - gamma_n * vrn, 0.0, None), 0.0)
            Fn_vec = Fn[..., None] * n
            vrt = vr - vrn[..., None] * n
            vt_mag = np.sqrt((vrt ** 2).sum(-1))
            Ft_mag = np.minimum(mu * Fn, gamma_n * vt_mag)
            Ft_dir = np.where(vt_mag[..., None] > 1e-10, vrt / (vt_mag[..., None] + 1e-12), 0.0)
            Ft_vec = -np.where(contact[..., None], Ft_mag[..., None] * Ft_dir, 0.0)
            forces = g[None, :] + (Fn_vec + Ft_vec).sum(axis=1)  # accel (unit mass)

            vel = vel + forces * dt_phys
            pos = pos + vel * dt_phys

            below = pos[:, 1] < radius
            pos[below, 1] = radius[below]; vel[below, 1] = np.maximum(vel[below, 1], 0.0)
            left = pos[:, 0] < radius; right = pos[:, 0] > L - radius
            pos[left, 0] = radius[left]; vel[left, 0] = np.maximum(vel[left, 0], 0.0)
            pos[right, 0] = L - radius[right]; vel[right, 0] = np.minimum(vel[right, 0], 0.0)

        pos_seq[t] = pos; vel_seq[t] = vel

    pos_seq /= L; vel_seq /= L
    pos3 = np.concatenate([pos_seq, np.zeros((T, N, 1))], axis=-1).astype(np.float32)
    vel3 = np.concatenate([vel_seq, np.zeros((T, N, 1))], axis=-1).astype(np.float32)
    node_type = np.zeros(N, dtype=np.int64)
    return pos3, vel3, radius.astype(np.float32), node_type


# ═══════════════════════════════════════════════════════════════════
# Graph collation (for PyG-style batching)
# ═══════════════════════════════════════════════════════════════════

def collate_graphs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    collated = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        collated[key] = torch.stack(values)
    return collated


# ═══════════════════════════════════════════════════════════════════
# Synthetic DEM data generator (vectorized, sub-stepped, cached)
# ═══════════════════════════════════════════════════════════════════

class SyntheticDEMData(Dataset):
    """Synthetic 2D granular column collapse for quick Trace testing.

    `dt` is the RECORDED timestep (what the trainer divides by for the accel
    target and what rollout integrates with). Physics integrates at
    `dt_phys = dt / substeps` for stability. No global velocity damping — energy
    is dissipated only by the physical contact dashpot, so the learnable
    dynamics aren't polluted by a non-physical global drag.
    """

    def __init__(
        self,
        n_samples: int = 200,
        n_particles: int = 400,
        n_steps: int = 120,
        dt: float = 0.003,
        box_size: float = 1.0,
        substeps: int = 20,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        n_jobs: Optional[int] = None,
        k_n: float = 2.0e5,
        gamma_n: float = 40.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.dt = dt
        self.box_size = box_size
        self.substeps = max(1, substeps)
        self.seed = seed
        self.cache_dir = cache_dir
        self.k_n = k_n
        self.gamma_n = gamma_n
        # Trajectories are independent -> generate them across CPU cores.
        self.n_jobs = n_jobs if n_jobs is not None else min(48, os.cpu_count() or 1)

        self.data = self._load_or_generate()

    # ------------------------------------------------------------------
    def _cache_path(self) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = (f"dem_N{self.n_particles}_T{self.n_steps}_dt{self.dt}"
               f"_sub{self.substeps}_kn{self.k_n:g}_gn{self.gamma_n:g}"
               f"_s{self.seed}_n{self.n_samples}.npz")
        return Path(self.cache_dir) / key

    def _load_or_generate(self) -> List[Dict[str, torch.Tensor]]:
        path = self._cache_path()
        if path is not None and path.exists():
            try:
                z = np.load(path)
                pos, vel, radius, ntype = z["pos"], z["vel"], z["radius"], z["node_type"]
                return [self._to_sample(pos[i], vel[i], radius[i], ntype[i])
                        for i in range(pos.shape[0])]
            except Exception:
                pass  # fall through to regeneration

        tasks = [(self.seed + i, self.n_particles, self.n_steps, self.box_size,
                  self.dt, self.substeps, self.k_n, self.gamma_n)
                 for i in range(self.n_samples)]
        n_jobs = max(1, min(self.n_jobs, self.n_samples))
        if n_jobs > 1:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_gen_traj, tasks)
        else:
            results = [_gen_traj(t) for t in tasks]

        pos_all, vel_all, rad_all, nt_all = [], [], [], []
        data = []
        for (p, v, r, nt) in results:
            pos_all.append(p); vel_all.append(v); rad_all.append(r); nt_all.append(nt)
            data.append(self._to_sample(p, v, r, nt))

        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp,
                         pos=np.stack(pos_all), vel=np.stack(vel_all),
                         radius=np.stack(rad_all), node_type=np.stack(nt_all))
                tmp.replace(path)  # atomic-ish; avoids partial files under DDP races
            except Exception:
                pass
        return data

    @staticmethod
    def _to_sample(pos, vel, radius, node_type) -> Dict[str, torch.Tensor]:
        return {
            "pos_seq": torch.tensor(pos, dtype=torch.float32),
            "vel_seq": torch.tensor(vel, dtype=torch.float32),
            "radius": torch.tensor(radius, dtype=torch.float32),
            "node_type": torch.tensor(node_type, dtype=torch.int64),
        }

    # ------------------------------------------------------------------
    def _generate_trajectory(self, seed: int):
        """Thin wrapper around the module-level worker (kept for compatibility)."""
        return _gen_traj((seed, self.n_particles, self.n_steps, self.box_size,
                          self.dt, self.substeps, self.k_n, self.gamma_n))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ═══════════════════════════════════════════════════════════════════
# Real DEM data loader (LIGGGHTS / Yade HDF5 output)
# ═══════════════════════════════════════════════════════════════════

class DEMDataset(Dataset):
    """Load real DEM simulation data from HDF5 files (see schema below)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        n_history: int = 4,
        include_contact_forces: bool = False,
        max_files: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) / split
        self.n_history = n_history
        self.include_contact_forces = include_contact_forces
        self.files = sorted(self.data_dir.glob("*.h5"))
        if max_files is not None:
            self.files = self.files[:max_files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.files[idx], "r") as f:
            sample = {
                "pos_seq": torch.tensor(f["pos"][:], dtype=torch.float32),
                "vel_seq": torch.tensor(f["vel"][:], dtype=torch.float32),
                "radius": torch.tensor(f["radius"][:], dtype=torch.float32),
                "node_type": torch.tensor(f["node_type"][:], dtype=torch.int64),
            }
            if self.include_contact_forces and "contact" in f:
                sample["edge_index_seq"] = torch.tensor(f["contact/edge_index"][:], dtype=torch.int64)
                sample["F_n_seq"] = torch.tensor(f["contact/F_n"][:], dtype=torch.float32)
                sample["F_t_seq"] = torch.tensor(f["contact/F_t"][:], dtype=torch.float32)
            if "material_id" in f:
                sample["material_id"] = torch.tensor(f["material_id"][:], dtype=torch.int64)
            return sample

    @staticmethod
    def create_dataloader(data_dir, split="train", batch_size=1, n_history=4,
                          include_contact_forces=False, num_workers=4, shuffle=True,
                          max_files=None) -> DataLoader:
        dataset = DEMDataset(data_dir, split, n_history, include_contact_forces, max_files)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=collate_graphs, pin_memory=True)
