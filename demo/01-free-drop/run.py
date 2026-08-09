"""
Run the trained TRACE model on a CUSTOM scene with no ground truth.

This is the minimal recipe for using TRACE as a forward simulator: build an
initial particle state yourself, hand it to `model.rollout`, and animate the
result. Here we drop a rectangular sand pile (about 1000 grains) with a chosen
initial velocity and let TRACE predict the collapse.

What YOU must provide (the "conditions" of the simulation):
  pos_0      (N, 2) initial particle positions, in the model's box frame [0, L]^2
  vel_0      (N, 2) initial velocities, as DISPLACEMENT PER FRAME (see note below)
  radius     (N,)   particle radius (same value the model was trained with)
  node_type  (N,)   particle type; sand is 0
  L, dt      the domain size and time step baked into the checkpoint

Everything else (contact graph, forces, memory, non-penetration) is handled by
the model. There is no reference trajectory and no loss; this is pure inference.

Units note — velocities are displacement per frame, not m/s. The model folds the
physical dt into 1.0 and reads velocity as v[t] = x[t] - x[t-1]. The training data
has a per-frame displacement std of about 0.0025 (see metadata vel_std), so a
"gentle" initial speed is a few thousandths of the box per frame and a "fast" drop
is around 0.01. Positions live in the shifted box [0, L], L = 0.8, floor at
y = radius, side walls at x in [radius, L - radius].

Usage:
  python demo/01-free-drop/run.py                      # default: 1000-grain drop
  python demo/01-free-drop/run.py --n 1500 --vy -0.012 --steps 300
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "trace"))
sys.path.insert(0, str(REPO / "common"))
from tracegnn.model2d import Trace2D                       # noqa: E402


# ── physical constants of the final 2D checkpoint ───────────────────────
CKPT = REPO / "checkpoints" / "sand2d" / "trace_2d_final.pt"
L = 0.8            # box size: domain is [0, L] x [0, L]
RADIUS = 0.0036    # particle radius the model was trained with
DT = 1.0           # model time step (physical dt is folded into 1.0)


def build_pile(n_target, cx, y_floor, vx, vy, seed=0):
    """A rectangular block of grains on a lattice, given a uniform initial velocity.

    Returns pos_0 (N,2) and vel_0 (N,2). N is close to n_target (snapped to a grid).
    """
    rng = np.random.default_rng(seed)
    spacing = 2.2 * RADIUS                                 # slightly more than one diameter
    # choose a roughly square block: cols x rows ~ n_target
    cols = int(round(np.sqrt(n_target)))
    rows = int(np.ceil(n_target / cols))
    width = (cols - 1) * spacing
    x0 = cx - width / 2.0
    xs = x0 + spacing * np.arange(cols)
    ys = y_floor + spacing * np.arange(rows)
    gx, gy = np.meshgrid(xs, ys)
    pos = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n_target]
    # tiny jitter so the lattice is not perfectly regular
    pos = pos + rng.normal(0.0, 0.15 * RADIUS, pos.shape)
    # keep everything inside the walls
    pos[:, 0] = np.clip(pos[:, 0], RADIUS + 1e-4, L - RADIUS - 1e-4)
    pos[:, 1] = np.clip(pos[:, 1], RADIUS + 1e-4, L - RADIUS - 1e-4)
    vel = np.tile(np.array([vx, vy], dtype=np.float32), (pos.shape[0], 1))
    return pos.astype(np.float32), vel.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="target particle count")
    ap.add_argument("--steps", type=int, default=300, help="rollout steps")
    ap.add_argument("--vx", type=float, default=0.0, help="initial x velocity (disp/frame)")
    ap.add_argument("--vy", type=float, default=-0.010, help="initial y velocity (disp/frame), negative = downward")
    ap.add_argument("--cx", type=float, default=0.40, help="pile center x")
    ap.add_argument("--y", type=float, default=0.45, help="pile bottom height")
    ap.add_argument("--gpu", type=str, default="0", help="CUDA device index, or 'cpu'")
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "free_drop.gif"))
    a = ap.parse_args()

    dev = torch.device("cpu") if a.gpu == "cpu" else torch.device(f"cuda:{a.gpu}")

    # ── 1. build the custom initial state (this is all the "input" the model needs) ──
    pos_np, vel_np = build_pile(a.n, a.cx, a.y, a.vx, a.vy)
    N = pos_np.shape[0]
    pos_0 = torch.from_numpy(pos_np).to(dev)
    vel_0 = torch.from_numpy(vel_np).to(dev)
    radius = torch.full((N,), RADIUS, device=dev)
    node_type = torch.zeros(N, dtype=torch.long, device=dev)     # sand = 0
    print(f"scene: {N} particles, initial velocity ({a.vx}, {a.vy}) disp/frame, {a.steps} steps")

    # ── 2. load the trained model from its checkpoint config ──
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    cfg = ck["config"]
    model = Trace2D(
        hidden_dim=int(cfg["hidden_dim"]),
        memory_dim=int(cfg["memory_dim"]),
        num_layers=int(cfg["num_layers"]),
        skin_factor=float(cfg["skin_factor"]),
        memory_type=cfg.get("memory", "st"),
        normalize_inputs=cfg.get("normalize_inputs", False),
        boundary_features=cfg.get("boundary_features", False),
    ).to(dev)
    if cfg.get("boundary_features"):
        model.box_size.fill_(L)
        model.feat_clip.fill_(float(cfg["skin_factor"]) * 2 * RADIUS)
    model.vel_from_displacement = bool(cfg.get("vel_from_displacement", False))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # ── 3. autoregressive rollout — no ground truth, pure forward simulation ──
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        traj = model.rollout(pos_0, vel_0, node_type, radius,
                             n_steps=a.steps, dt=DT, box_size=L,
                             contact_projection=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0
    frames = np.stack([pos_0.cpu().numpy()] + [t["pos"].cpu().numpy() for t in traj], 0)
    print(f"rollout done: {frames.shape[0]} frames | "
          f"inference {t_infer:.2f} s for {a.steps} steps "
          f"({1000*t_infer/a.steps:.1f} ms/step, {N} particles)")

    # ── 4. animate ──
    render_gif(frames, a.out)
    print(f"saved animation -> {a.out}")


def render_gif(frames, out_path, fps=20, dt_phys=0.0025):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    color = frames[0, :, 1]                                # color by initial height
    T = len(frames) - 1
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("TRACE — custom sand drop", fontsize=12)
    sc = ax.scatter(frames[0, :, 0], frames[0, :, 1], c=color, cmap="viridis", s=8)
    step_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=11,
                       va="top", ha="left", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6))

    def update(i):
        sc.set_offsets(frames[i])
        step_txt.set_text(f"step {i:3d}/{T}\nt = {i * dt_phys:.3f} s")
        return sc, step_txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
