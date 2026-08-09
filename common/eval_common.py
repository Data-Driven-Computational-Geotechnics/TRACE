"""Comprehensive rollout evaluation for granular dynamics simulators.

Metrics (per-step, averaged over test trajectories):
  1. Position RMSE        — are particles where they should be?
  2. Kinetic Energy       — does energy decay at the right rate?
  3. Overlap ratio        — are particles unphysically interpenetrating?
  4. Momentum residual     — does Newton's 3rd law hold in practice?

Summary metrics:
  5. Final deposit shape  — runout distance, pile height (engineering relevance)
  6. Long-term KE         — does the system settle to rest or diverge?

Outputs (saved to <save_dir>/):
  eval_metrics.json        — all numbers, machine-readable
  eval_rmse.png            — RMSE vs rollout step
  eval_energy.png          — Kinetic Energy (pred vs GT) vs step
  eval_overlap.png         — Overlap ratio vs step
  eval_deposit.png         — Final frame: GT vs predicted particle positions
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
#  single-trajectory metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_rmse(pred_pos, gt_pos):
    """Per-step position RMSE: sqrt(mean(||pred - gt||²)).  Returns (n_steps,)."""
    sq = ((pred_pos - gt_pos) ** 2).sum(dim=-1)           # (T, N)
    return sq.mean(dim=-1).sqrt().cpu().numpy()            # (T,)


def _compute_ke(vel, mass=1.0):
    """Per-step kinetic energy: 0.5 * m * sum(v²).  Returns (n_steps,)."""
    return 0.5 * mass * (vel ** 2).sum(dim=(-2, -1)).cpu().numpy()


def _compute_overlap_ratio(pos, radius, box_size):
    """Fraction of particles with any overlap beyond a tolerance.

    For each particle i, find its nearest neighbour j and compute
        overlap = r_i + r_j - dist(i,j).
    If overlap > tol, the particle is in penetration.
    Returns a scalar ∈ [0, 1].
    """
    N = pos.shape[0]
    if N < 2:
        return 0.0
    tol = 1e-6 * box_size
    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)  # (N, N)
    # mask out self-distance
    dist = dist + torch.eye(N, device=pos.device, dtype=pos.dtype) * 1e9
    r_sum = radius.unsqueeze(0) + radius.unsqueeze(1)                 # (N, N)
    overlap = r_sum - dist                                              # (N, N): >0 = penetration
    min_overlap_per_particle = overlap.min(dim=1).values                # min overlap for each i
    penetrating = (min_overlap_per_particle > tol).float().mean().item()
    return penetrating


def _compute_momentum_residual(pred_vel, gt_vel, acc_ext=None):
    """Internal-force momentum conservation residual.

    Δp_total = Σ m_i * Δv_i.  For a closed system with no external forces,
    Δp should be zero.  Gravity/box reactions create a non-zero external component;
    we report the magnitude as a diagnostic.  Units: kg·m/s per particle.
    Returns scalar.
    """
    dp = (pred_vel[-1] - pred_vel[0]) - (gt_vel[-1] - gt_vel[0])  # (N, dim)
    return dp.norm(dim=-1).mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
#  main evaluation entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(model, samples, dev, dim, dt, box_size, radius,
             n_rollout=120, n_long=240, save_dir=None):
    """Run full evaluation on a list of trajectory dicts (SandH5Data format).

    Parameters
    ----------
    model : nn.Module
        Trained TRACE/GNS/TGNNS model (must have a .rollout method).
    samples : list[dict]
        List of {pos, vel, radius, node_type} dicts (SandH5Data format).
    dev : torch.device
    dim : int
        Spatial dimension (2 or 3).
    dt : float
        Time step (GNS convention: dt=1 for displacement-based velocity).
    box_size : float
        Domain side length.
    radius : float or torch.Tensor
        Per-particle or scalar radius.
    n_rollout : int
        Steps for per-step metrics (RMSE, KE, overlap).
    n_long : int
        Steps for the long-horizon KE stability check.
    save_dir : Path or None
        Where to save eval_metrics.json and eval_*.png figures.

    Returns
    -------
    dict with keys: rmse, ke_pred, ke_gt, overlap, deposit, long_ke, mom_resid
    """
    model.eval()
    n_traj = len(samples)

    # accumulators (list of arrays, one per trajectory)
    all_rmse = []
    all_ke_pred = []
    all_ke_gt = []
    all_overlap = []
    deposits = []
    long_kes = []
    mom_resids = []

    with torch.no_grad():
        for s in samples:
            pos_gt_all = torch.as_tensor(s["pos"]).to(dev)
            vel_all = torch.as_tensor(s["vel"]).to(dev)
            nt = torch.as_tensor(s["node_type"]).to(dev)
            rad = torch.as_tensor(s["radius"]).to(dev)
            if rad.ndim == 0:
                rad = rad.expand(pos_gt_all.shape[1])

            # ★从 frame1 用真实初速度 vel[1] 播种★(vel[0]=0 会丢初动量→自由落体, 见 train.py 注释)
            R = min(n_rollout, pos_gt_all.shape[0] - 2)
            R_long = min(n_long, pos_gt_all.shape[0] - 2)

            # ── rollout (short) ──
            traj = model.rollout(pos_gt_all[1], vel_all[1], nt, rad,
                                 n_steps=R, dt=dt, box_size=box_size,
                                 contact_projection=True)
            pred_pos = torch.stack([t["pos"] for t in traj], 0)     # (R, N, dim) frames 2..R+1
            pred_vel = torch.stack([t["vel"] for t in traj], 0)
            gt_pos = pos_gt_all[2:R + 2]
            gt_vel = vel_all[2:R + 2]

            all_rmse.append(_compute_rmse(pred_pos, gt_pos))
            all_ke_pred.append(_compute_ke(pred_vel))
            all_ke_gt.append(_compute_ke(gt_vel))
            all_overlap.append(np.array([
                _compute_overlap_ratio(pred_pos[t], rad, box_size)
                for t in range(R)]))

            # ── final deposit metrics ──
            # runout: 2D 用 x 轴延展; 3D 取两个水平轴(x,z)延展的较大者(主铺展方向)
            def _runout(p):
                ext = p[-1, :, 0].max() - p[-1, :, 0].min()
                if dim == 3:
                    ext = torch.maximum(ext, p[-1, :, 2].max() - p[-1, :, 2].min())
                return ext.item()
            deposits.append({
                "runout_pred": _runout(pred_pos),
                "runout_gt": _runout(gt_pos),
                "height_pred": pred_pos[-1, :, 1].max().item(),
                "height_gt": gt_pos[-1, :, 1].max().item(),
            })

            # ── momentum residual ──
            mom_resids.append(_compute_momentum_residual(pred_vel, gt_vel))

            # ── long-rollout KE ──
            traj_long = model.rollout(pos_gt_all[1], vel_all[1], nt, rad,
                                      n_steps=R_long, dt=dt, box_size=box_size,
                                      contact_projection=True)
            pred_vel_long = torch.stack([t["vel"] for t in traj_long], 0)
            long_kes.append(_compute_ke(pred_vel_long))

    # ── aggregate across trajectories ──
    # align to same length (take min)
    min_len = min(len(a) for a in all_rmse)
    all_rmse = np.array([a[:min_len] for a in all_rmse])
    all_ke_pred = np.array([a[:min_len] for a in all_ke_pred])
    all_ke_gt = np.array([a[:min_len] for a in all_ke_gt])
    all_overlap = np.array([a[:min_len] for a in all_overlap])
    long_kes = np.array([a for a in long_kes])  # may differ in length, keep as list of arrays
    mom_resids = np.array(mom_resids)

    rmse_mean = all_rmse.mean(axis=0)
    rmse_std = all_rmse.std(axis=0)
    ke_pred_mean = all_ke_pred.mean(axis=0)
    ke_gt_mean = all_ke_gt.mean(axis=0)
    overlap_mean = all_overlap.mean(axis=0)

    deposit = {
        "runout_pred_mean": np.mean([d["runout_pred"] for d in deposits]),
        "runout_gt_mean": np.mean([d["runout_gt"] for d in deposits]),
        "height_pred_mean": np.mean([d["height_pred"] for d in deposits]),
        "height_gt_mean": np.mean([d["height_gt"] for d in deposits]),
    }
    final_ke_pred = np.mean([ke[-1] for ke in long_kes])
    mom_resid_mean = float(mom_resids.mean())

    summary = {
        "n_trajectories": n_traj,
        "n_rollout": min_len,
        "n_long": n_long,
        "rmse_end": float(rmse_mean[-1]),
        "ke_pred_end": float(ke_pred_mean[-1]),
        "ke_gt_end": float(ke_gt_mean[-1]),
        "overlap_max": float(overlap_mean.max()),
        "overlap_end": float(overlap_mean[-1]),
        "long_ke_final": float(final_ke_pred),
        "mom_resid": mom_resid_mean,
        "deposit": deposit,
    }

    # ── save outputs ──
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        json.dump(summary, open(save_dir / "eval_metrics.json", "w"), indent=2,
                  default=float)

        _plot_rmse(rmse_mean, rmse_std, min_len, save_dir)
        _plot_energy(ke_pred_mean, ke_gt_mean, min_len, save_dir)
        _plot_overlap(overlap_mean, min_len, save_dir)
        _plot_deposit(pred_pos[-1].cpu().numpy(), gt_pos[-1].cpu().numpy(),
                      deposits, box_size, save_dir, dim=dim)

    return {
        "rmse": rmse_mean, "rmse_std": rmse_std,
        "ke_pred": ke_pred_mean, "ke_gt": ke_gt_mean,
        "overlap": overlap_mean, "deposit": deposit,
        "long_ke": final_ke_pred, "mom_resid": mom_resid_mean,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_rmse(rmse_mean, rmse_std, n_steps, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(n_steps)
    ax.plot(steps, rmse_mean, color="#c0392b", lw=1.5)
    ax.fill_between(steps, rmse_mean - rmse_std, rmse_mean + rmse_std,
                    color="#c0392b", alpha=0.15)
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Position RMSE")
    ax.set_title(f"Position RMSE vs Rollout Step  (end={rmse_mean[-1]:.3e})",
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_dir / "eval_rmse.png", dpi=150); plt.close(fig)


def _plot_energy(ke_pred, ke_gt, n_steps, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(n_steps)
    ax.plot(steps, ke_gt, color="#2c3e50", lw=1.5, label="Reference (GT)")
    ax.plot(steps, ke_pred, color="#c0392b", lw=1.5, label="TRACE")
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Kinetic Energy")
    ax.set_title("Kinetic Energy Evolution", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_dir / "eval_energy.png", dpi=150); plt.close(fig)


def _plot_overlap(overlap, n_steps, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(n_steps)
    ax.plot(steps, overlap * 100, color="#e67e22", lw=1.5)
    ax.axhline(5, color="#999", ls="--", lw=0.8, label="5% threshold")
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Overlap ratio (%)")
    ax.set_title(f"Particle Penetration  (max={overlap.max()*100:.1f}%, end={overlap[-1]*100:.1f}%)",
                 fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_dir / "eval_overlap.png", dpi=150); plt.close(fig)


def _plot_deposit(pred_final, gt_final, deposits, box_size, save_dir, dim=2):
    if dim == 3:
        # 3D: 正视(x-y, 压扁z) + 俯视(x-z, 压扁y) 双视图, 2x2
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        views = [("front (x-y)", 0, 1, box_size * 0.5), ("top (x-z)", 0, 2, box_size)]
        for r, (vname, ax_a, ax_b, ylim) in enumerate(views):
            for c, (pos, label) in enumerate([(gt_final, "Reference (GT)"), (pred_final, "TRACE")]):
                ax = axes[r, c]
                ax.scatter(pos[:, ax_a], pos[:, ax_b], s=1, alpha=0.4,
                           c="#2c3e50" if "GT" in label else "#c0392b")
                ax.set_xlim(0, box_size); ax.set_ylim(0, ylim)
                ax.set_aspect("equal"); ax.set_title(f"{label} — {vname}", fontweight="bold")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for ax, pos, label in [(ax1, gt_final, "Reference (GT)"), (ax2, pred_final, "TRACE")]:
            ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.6, c="#2c3e50" if "GT" in label else "#c0392b")
            ax.set_xlim(0, box_size); ax.set_ylim(0, box_size * 0.5)
            ax.set_aspect("equal"); ax.set_title(label, fontweight="bold")

    # aggregate deposit stats across trajectories
    rp = np.mean([d["runout_pred"] for d in deposits])
    rg = np.mean([d["runout_gt"] for d in deposits])
    hp = np.mean([d["height_pred"] for d in deposits])
    hg = np.mean([d["height_gt"] for d in deposits])
    fig.suptitle(
        f"Final Deposit  |  runout: {rp:.3f} vs GT {rg:.3f}  |  height: {hp:.3f} vs GT {hg:.3f}",
        fontsize=10)
    fig.tight_layout(); fig.savefig(save_dir / "eval_deposit.png", dpi=150); plt.close(fig)
