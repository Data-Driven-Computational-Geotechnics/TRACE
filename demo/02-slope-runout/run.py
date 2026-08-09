"""
Custom scene: a sand pile slides down a slope and spreads on the flat runout.

The trained model only knows a flat-floored box, so the slope is not a model
boundary. We build it from a dense band of FIXED "wall" particles and run a
custom rollout that re-freezes those particles every step. The moving sand feels
the slope through the learned repulsive contact forces, exactly as it feels any
other grain. This is an out-of-distribution use of the model (it was trained on
flat column collapse), so treat the result as a qualitative demonstration.

Scene:
  - a ramp descending from the upper left to a flat section on the right,
    made of frozen wall particles (drawn in gray);
  - a rectangular sand pile resting on the upper ramp (colored by height);
  - gravity, learned by the model, pulls the pile down the incline; it slides,
    reaches the horizontal runout, and spreads.

Usage:
  /data/envs/trace/bin/python demo/run_slope_scene.py --gpu 0
  python demo/run_slope_scene.py --gpu cpu
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

CKPT = REPO / "checkpoints" / "sand2d" / "trace_2d_final.pt"
L = 0.8
RADIUS = 0.0036
DT = 1.0


# slope geometry: descends from the heel (upper left) to the toe, then a long flat runout
P_HEEL = np.array([0.04, 0.62])
P_TOE = np.array([0.34, 0.05])
_DV = P_TOE - P_HEEL
_LEN = np.linalg.norm(_DV)
TANGENT = (_DV / _LEN).astype(np.float32)                  # downhill unit vector along the slope
NORMAL = np.array([-TANGENT[1], TANGENT[0]], dtype=np.float32)  # unit normal pointing away from the surface
Y_FLAT = float(P_TOE[1])


def slope_surface(x):
    """Surface height of the ramp at horizontal position x (heel -> toe, then flat)."""
    y = np.where(
        x <= P_HEEL[0], P_HEEL[1],
        np.where(x >= P_TOE[0], Y_FLAT,
                 P_HEEL[1] + (Y_FLAT - P_HEEL[1]) * (x - P_HEEL[0]) / (P_TOE[0] - P_HEEL[0])))
    return y


def build_slope(bands=9):
    """A thick band of fixed particles under the ramp surface. Returns (M, 2)."""
    d = 1.7 * RADIUS                                       # dense packing
    xs = np.arange(0.05, 0.78, d)
    pts = []
    for x in xs:
        top = slope_surface(np.array([x]))[0]
        for b in range(bands):                             # stack downward from the surface
            pts.append((x, top - b * d))
    return np.array(pts, dtype=np.float32)


def build_pile(n_target, along=0.30, length=0.34, thick=0.13, gap=0.02, seed=0):
    """A small rectangular block ALIGNED with the slope, resting on the upper ramp.

    The block's long axis follows the downhill tangent and its short axis follows
    the surface normal, so it lies flat against the incline. `along` is the
    fraction of the slope (heel -> toe) where the block is centered.
    """
    rng = np.random.default_rng(seed)
    d = 2.2 * RADIUS
    c_surf = P_HEEL + along * _DV                          # point on the surface
    center = c_surf + NORMAL * (gap + thick / 2.0)         # lift the block off the surface
    na = max(2, int(round(length / d)))                    # grains along the slope
    nb = max(2, int(round(thick / d)))                     # grains across the thickness
    a = (np.arange(na) - (na - 1) / 2.0) * d               # tangential offsets
    b = (np.arange(nb) - (nb - 1) / 2.0) * d               # normal offsets
    ga, gb = np.meshgrid(a, b)
    pts = center + ga.ravel()[:, None] * TANGENT + gb.ravel()[:, None] * NORMAL
    pts = pts[:n_target] if n_target < pts.shape[0] else pts
    pts = pts + rng.normal(0.0, 0.15 * RADIUS, pts.shape)
    return pts.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="max sand particle count")
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--speed", type=float, default=0.007, help="initial downhill speed along the slope (disp/frame)")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "slope_runout.gif"))
    a = ap.parse_args()
    dev = torch.device("cpu") if a.gpu == "cpu" else torch.device(f"cuda:{a.gpu}")

    # ── build the scene: fixed slope + a slope-aligned sand block ──
    wall = build_slope()
    sand = build_pile(a.n)
    M, S = wall.shape[0], sand.shape[0]
    pos_np = np.concatenate([wall, sand], axis=0)
    N = pos_np.shape[0]
    frozen = np.zeros(N, dtype=bool)
    frozen[:M] = True                                      # first M particles are the slope
    vel_np = np.zeros((N, 2), dtype=np.float32)
    vel_np[M:] = a.speed * TANGENT                         # initial velocity along the downhill tangent
    print(f"scene: {S} sand grains (slope-aligned block) on a {M}-particle slope, {a.steps} steps")

    pos = torch.from_numpy(pos_np).to(dev)
    vel = torch.from_numpy(vel_np).to(dev)
    radius = torch.full((N,), RADIUS, device=dev)
    node_type = torch.zeros(N, dtype=torch.long, device=dev)
    frozen_t = torch.from_numpy(frozen).to(dev)
    pos_fix = pos[frozen_t].clone()

    # ── load the trained model ──
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    cfg = ck["config"]
    model = Trace2D(
        hidden_dim=int(cfg["hidden_dim"]), memory_dim=int(cfg["memory_dim"]),
        num_layers=int(cfg["num_layers"]), skin_factor=float(cfg["skin_factor"]),
        memory_type=cfg.get("memory", "st"),
        normalize_inputs=cfg.get("normalize_inputs", False),
        boundary_features=cfg.get("boundary_features", False)).to(dev)
    if cfg.get("boundary_features"):
        model.box_size.fill_(L)
        model.feat_clip.fill_(float(cfg["skin_factor"]) * 2 * RADIUS)
    model.vel_from_displacement = bool(cfg.get("vel_from_displacement", False))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # ── custom rollout that re-freezes the slope particles every step ──
    memory, id_map = None, None
    frames = [pos.cpu().numpy().copy()]
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(a.steps):
            res = model.forward(pos, vel, node_type, radius,
                                edge_memory_state=memory, edge_id_map=id_map,
                                training_noise=False)
            prev = pos
            vel = vel + res["accel_phys"] * DT
            pos = pos + vel * DT
            pos, vel = model._constrain_state(pos, vel, prev, radius, L, DT,
                                              contact_projection=True, proj_iters=25)
            pos[frozen_t] = pos_fix                        # slope stays put
            vel[frozen_t] = 0.0
            memory, id_map = res["edge_memory_state"], res["edge_id_map"]
            frames.append(pos.cpu().numpy().copy())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0
    frames = np.stack(frames, 0)
    print(f"rollout done: {frames.shape[0]} frames | "
          f"inference {t_infer:.2f} s for {a.steps} steps "
          f"({1000*t_infer/a.steps:.1f} ms/step, {N} particles incl. {M} slope)")

    render_gif(frames, frozen, a.out)
    print(f"saved animation -> {a.out}")


def render_gif(frames, frozen, out_path, fps=25, dt_phys=0.0025):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sand = ~frozen
    color = frames[0, sand, 1]                              # color sand by initial height (as in assets)
    # crop the empty upper part of the box so the wide slope+runout fills the frame
    y_hi = float(max(frames[:, :, 1].max(), P_HEEL[1])) + 0.03
    T = len(frames) - 1

    fig, ax = plt.subplots(figsize=(7.4, 7.4 * y_hi / L), dpi=150)
    ax.set_xlim(0, L); ax.set_ylim(0, y_hi)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("TRACE — sand runout on a slope", fontsize=12)
    ax.scatter(frames[0, frozen, 0], frames[0, frozen, 1], c="0.62", s=6)   # static slope
    sc = ax.scatter(frames[0, sand, 0], frames[0, sand, 1], c=color, cmap="viridis", s=7)
    step_txt = ax.text(0.015, 0.965, "", transform=ax.transAxes, fontsize=11,
                       va="top", ha="left", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6))

    def update(i):
        sc.set_offsets(frames[i, sand])
        step_txt.set_text(f"step {i:3d}/{T}\nt = {i * dt_phys:.3f} s")
        return sc, step_txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
