"""Side-by-side ground-truth vs model-rollout rendering for granular sims.

Produces, for a SINGLE trajectory:
  * a static multi-frame panel  (row 1: DEM ground truth, row 2: model rollout;
    columns are selected timesteps, front-loaded onto the fast collapse phase)
  * an animated GIF             (DEM | model, side by side)

Shared by all three method packages (trace / gns / tgnn) via
``tools/render_rollout.py``. The model only needs a ``.rollout(pos0, vel0,
node_type, radius, n_steps, dt, box_size)`` method returning a list of per-step
dicts each carrying a ``"pos"`` key — the SandH5Data convention used everywhere
in this repo. Positions are assumed to live in the shifted box ``[0, box_size]``.
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
# 中文字体: 系统装了 Noto Sans CJK SC, 否则中文标题/标签会渲染成空方框(豆腐块)
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ── frame selection ────────────────────────────────────────────────────────────

def _pick_frames(n_steps, k=6):
    """Front-loaded frame indices in [0, n_steps].

    Granular collapse is fast (peak motion within the first ~15% of the
    trajectory, static thereafter), so we sample the early dynamics densely and
    the static tail sparsely instead of spacing frames evenly.
    """
    fracs = [0.0, 0.05, 0.15, 0.35, 0.65, 1.0][:k]
    idx = sorted({int(round(f * n_steps)) for f in fracs})
    return idx


# ── rollout ─────────────────────────────────────────────────────────────────────

def rollout_pred(model, sample, dev, dt, box_size, n_steps):
    """Run the model autoregressively; return (pred_pos, gt_pos) as (R+1, N, d).

    ★Seeded from frame 1 with the TRUE initial velocity vel[1]★ (frame 1 is the
    first frame with a real velocity; vel[0] is zeroed by the loader, and seeding
    from it discards each block's initial momentum/direction → the rollout
    degenerates to free-fall). pred/gt share frame 1 as the initial condition and
    are aligned frame-for-frame from there.
    """
    pos = torch.as_tensor(sample["pos"]).to(dev)
    vel = torch.as_tensor(sample["vel"]).to(dev)
    nt = torch.as_tensor(sample["node_type"]).to(dev)
    rad = torch.as_tensor(sample["radius"]).to(dev)
    R = min(n_steps, pos.shape[0] - 2)
    model.eval()
    with torch.no_grad():
        traj = model.rollout(pos[1], vel[1], nt, rad, n_steps=R, dt=dt,
                             box_size=box_size, contact_projection=True)
    pred = torch.stack([pos[1]] + [t["pos"] for t in traj], 0)   # frames 1..R+1
    gt = pos[1: R + 2]                                           # frames 1..R+1
    return pred.cpu().numpy(), gt.cpu().numpy()


def _perstep_rmse(pred, gt):
    """Per-frame position RMSE, (T,)."""
    return np.sqrt(((pred - gt) ** 2).sum(-1).mean(-1))


# ── static panel ────────────────────────────────────────────────────────────────

def render_panel(pred, gt, box_size, save_path, title="", frames=None):
    """2 rows (GT / prediction) x len(frames) columns; particles colored by
    initial height so the same grain keeps its color across both rows."""
    T = gt.shape[0]
    frames = frames if frames is not None else _pick_frames(T - 1)
    frames = [min(f, T - 1) for f in frames]
    rmse = _perstep_rmse(pred, gt)
    color = gt[0, :, 1]                                          # initial height

    ncol = len(frames)
    fig, axes = plt.subplots(2, ncol, figsize=(2.4 * ncol, 5.2))
    if ncol == 1:
        axes = axes.reshape(2, 1)
    row_label = ["Reference (ground truth)", "model rollout"]
    for r, data in enumerate((gt, pred)):
        for c, f in enumerate(frames):
            ax = axes[r, c]
            ax.scatter(data[f, :, 0], data[f, :, 1], c=color, cmap="viridis",
                       s=4, edgecolors="none")
            ax.set_xlim(0, box_size); ax.set_ylim(0, box_size)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={f}" + (f"\nRMSE={rmse[f]:.2e}" if f > 0 else ""),
                             fontsize=9)
            if c == 0:
                ax.set_ylabel(row_label[r], fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rmse


# ── animation ───────────────────────────────────────────────────────────────────

def render_gif(pred, gt, box_size, save_path, stride=2, fps=25, title=""):
    """DEM | model side-by-side animation over the whole rollout."""
    T = gt.shape[0]
    color = gt[0, :, 1]
    frames = list(range(0, T, stride))
    fig, (axg, axp) = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, name in ((axg, "Reference (ground truth)"), (axp, "model rollout")):
        ax.set_xlim(0, box_size); ax.set_ylim(0, box_size)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=11, fontweight="bold")
    sg = axg.scatter(gt[0, :, 0], gt[0, :, 1], c=color, cmap="viridis", s=5, edgecolors="none")
    sp = axp.scatter(pred[0, :, 0], pred[0, :, 1], c=color, cmap="viridis", s=5, edgecolors="none")
    sup = fig.suptitle(f"{title}  t=0", fontsize=11, fontweight="bold")

    def update(f):
        sg.set_offsets(gt[f, :, :2])
        sp.set_offsets(pred[f, :, :2])
        sup.set_text(f"{title}  t={f}")
        return sg, sp, sup

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


# ── one-call entry point ────────────────────────────────────────────────────────

def render_comparison(model, sample, dev, dt, box_size, n_steps, save_dir,
                      name="rollout", make_gif=True, gif_stride=2):
    """Run rollout + write panel.png, anim.gif and perstep_rmse.npy into save_dir.

    Returns the per-step RMSE array.
    """
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    pred, gt = rollout_pred(model, sample, dev, dt, box_size, n_steps)
    rmse = render_panel(pred, gt, box_size, save_dir / f"{name}_panel.png",
                        title=f"{name}  |  {gt.shape[0]-1}-step rollout  "
                              f"({gt.shape[1]} particles)")
    np.save(save_dir / f"{name}_perstep_rmse.npy", rmse)
    if make_gif:
        render_gif(pred, gt, box_size, save_dir / f"{name}_anim.gif",
                   stride=gif_stride, title=name)
    return rmse
