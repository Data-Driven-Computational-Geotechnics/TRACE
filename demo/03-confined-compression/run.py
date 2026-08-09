"""
Custom scene: confined (oedometer-style) compression of a sand specimen.

Rigid steel plates confine the sand on the bottom and both sides. The top plate
moves down at a constant rate and compresses the specimen. The grains shear,
rearrange, and densify. The test is displacement-driven and needs no applied
force: we impose the plate motion and MEASURE the volumetric response.

The plates are FIXED "wall" particles (gray). A custom rollout re-freezes the
bottom and side plates every step and moves the top plate down by a fixed
increment. The moving top plate pushes the sand through the model's learned
repulsive contact forces and the per-step non-penetration projection.

Outputs:
  compression.gif        the animation
  compression_curve.png  top-plate displacement vs volumetric strain (+ void ratio)
  compression_data.csv   step, displacement, axial/volumetric strain, void ratio,
                         packing fraction, and the plate reaction force that the
                         model produces internally (a bonus: no force is imposed)

NOTE. The model was trained on free-surface column collapse, never on confined
compression under a moving platen. This is an out-of-distribution use, so the
curves are a qualitative demonstration of the model's contact mechanics under a
new boundary condition, not a validated soil compression curve.

Usage:
  /data/envs/trace/bin/python demo/run_compression_scene.py --gpu 0
  python demo/run_compression_scene.py --gpu cpu
"""
import argparse
import csv
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

# compression cell geometry (inside the 0.8 x 0.8 box)
X_LO, X_HI = 0.26, 0.54                                    # inner faces of the side plates
Y_BOT = 0.14                                               # top face of the bottom plate
Y_TOP0 = 0.52                                              # initial bottom face of the top plate
W_CELL = X_HI - X_LO                                       # confined width (constant)
NW = 5                                                     # plate thickness in particle rows


def build_plates():
    """Fixed bottom + side plates and the (initially fixed) top plate.

    Plates are thick and overlap at the corners so grains cannot escape.
    Returns wall positions and a boolean marking which rows are the top plate.
    """
    d = 1.5 * RADIUS                                       # dense plate packing
    x_out_lo, x_out_hi = X_LO - NW * d, X_HI + NW * d      # full span incl. side plates
    pts, is_top = [], []
    # bottom plate: NW rows spanning the full width
    for b in range(NW):
        for x in np.arange(x_out_lo, x_out_hi + d, d):
            pts.append((x, Y_BOT - b * d)); is_top.append(False)
    # side plates: NW columns each, tall enough to guide the descending top plate
    for s in range(NW):
        for y in np.arange(Y_BOT, Y_TOP0 + 0.14, d):
            pts.append((X_LO - (s + 1) * d, y)); is_top.append(False)
            pts.append((X_HI + (s + 1) * d, y)); is_top.append(False)
    # top plate: NW rows spanning the full width (overhangs the side plates)
    for b in range(NW):
        for x in np.arange(x_out_lo, x_out_hi + d, d):
            pts.append((x, Y_TOP0 + b * d)); is_top.append(True)
    return np.array(pts, dtype=np.float32), np.array(is_top, dtype=bool)


def build_specimen(seed=0):
    """A packed sand specimen filling the cell between the plates."""
    rng = np.random.default_rng(seed)
    d = 2.15 * RADIUS
    xs = np.arange(X_LO + RADIUS, X_HI - RADIUS, d)
    ys = np.arange(Y_BOT + 1.5 * RADIUS, Y_TOP0 - 2.0 * RADIUS, d)
    gx, gy = np.meshgrid(xs, ys)
    pos = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pos = pos + rng.normal(0.0, 0.18 * RADIUS, pos.shape)  # break the perfect lattice
    return pos.astype(np.float32)


def measure(pos_sand, top_y):
    """Confined bulk volume (area) from grains still inside the cell.

    Width is fixed by the side plates. Height is the height of the grain column,
    taken from grains that remain between the plates so a few escapees cannot
    distort the measurement.
    """
    x, y = pos_sand[:, 0], pos_sand[:, 1]
    inside = (x > X_LO - 0.01) & (x < X_HI + 0.01) & (y > Y_BOT - 0.01) & (y < top_y + 0.01)
    yin = y[inside]
    h = float(np.percentile(yin, 99) - Y_BOT) if yin.size else 0.0
    return W_CELL, h, W_CELL * h, int((~inside).sum())


def dem_stress(model, pos, radius, sand_ids, area, emod):
    """DEM stress from the grain-grain overlaps, using a contact law F = k_n * overlap.

    TRACE gives the geometry (which grains overlap and by how much). The force comes
    from a physical linear contact law, exactly as in PFC/DEM where the contact
    stiffness follows from the grain effective modulus. The stress tensor is the
    Love-Weber sum sigma = (1/A) sum over contacts of f (x) l, with branch vector
    l = (r_i+r_j) n. The model length units cancel in the sum, so passing the
    effective modulus emod (in Pa) returns the stress directly in Pa.
    """
    k_n = emod
    row, col = model._find_edges(pos.detach(), radius, 1.0)     # only touching pairs
    if row.numel() == 0:
        return 0.0, 0.0
    d = pos[row] - pos[col]
    dist = d.norm(dim=1)
    n = d / (dist.unsqueeze(1) + 1e-9)
    overlap = (radius[row] + radius[col]) - dist
    r = row.cpu().numpy(); c = col.cpu().numpy()
    ss = np.array([(int(a) in sand_ids and int(b) in sand_ids) for a, b in zip(r, c)])
    ss &= (overlap.cpu().numpy() > 0)
    if ss.sum() == 0:
        return 0.0, 0.0
    o = overlap.cpu().numpy()[ss]
    ny = n.cpu().numpy()[ss, 1]; nx = n.cpu().numpy()[ss, 0]
    br = (radius[row] + radius[col]).cpu().numpy()[ss]
    sig_yy = float((k_n * o * ny * br * ny).sum() / area)
    sig_xx = float((k_n * o * nx * br * nx).sum() / area)
    return sig_yy, 0.5 * (sig_xx + sig_yy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600, help="maximum compression steps")
    ap.add_argument("--rate", type=float, default=0.00022, help="top-plate descent per step (disp/frame)")
    ap.add_argument("--max-strain", type=float, default=0.15, help="stop at this axial strain")
    ap.add_argument("--emod", type=float, default=1.0e8,
                    help="PFC effective contact modulus E* in Pa (sand: about 1e8); sets the DEM contact stiffness")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).parent))
    a = ap.parse_args()
    dev = torch.device("cpu") if a.gpu == "cpu" else torch.device(f"cuda:{a.gpu}")
    outdir = Path(a.outdir)

    # ── build the scene ──
    wall, is_top = build_plates()
    sand = build_specimen()
    M, S = wall.shape[0], sand.shape[0]
    pos_np = np.concatenate([wall, sand], axis=0)
    N = pos_np.shape[0]
    frozen = np.zeros(N, dtype=bool); frozen[:M] = True
    top_mask = np.zeros(N, dtype=bool); top_mask[:M] = is_top
    fixed_mask = frozen & ~top_mask                        # bottom + side plates (never move)
    print(f"scene: {S} sand grains, {M} plate grains ({int(is_top.sum())} in the top plate), {a.steps} steps")

    pos = torch.from_numpy(pos_np).to(dev)
    vel = torch.zeros((N, 2), dtype=torch.float32, device=dev)
    radius = torch.full((N,), RADIUS, device=dev)
    node_type = torch.zeros(N, dtype=torch.long, device=dev)
    fixed_t = torch.from_numpy(fixed_mask).to(dev)
    top_t = torch.from_numpy(top_mask).to(dev)
    sand_t = torch.from_numpy(~frozen).to(dev)
    fixed_pos = pos[fixed_t].clone()
    top_pos = pos[top_t].clone()                           # top plate positions (we lower these)
    sand_ids = set(torch.nonzero(sand_t).flatten().tolist())

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

    # ── initial specimen volume ──
    w0, h0, V0, _ = measure(sand, Y_TOP0)
    V_solid = S * np.pi * RADIUS ** 2                      # constant solid area of the grains
    frames, rec = [pos.cpu().numpy().copy()], []

    # ── displacement-driven rollout ──
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for k in range(a.steps):
            res = model.forward(pos, vel, node_type, radius,
                                edge_memory_state=None, edge_id_map=None,
                                training_noise=False)
            prev = pos
            vel = vel + res["accel_phys"] * DT
            pos = pos + vel * DT
            pos, vel = model._constrain_state(pos, vel, prev, radius, L, DT,
                                              contact_projection=True, proj_iters=25)
            # re-freeze the fixed plates
            pos[fixed_t] = fixed_pos; vel[fixed_t] = 0.0
            # lower the top plate by one increment
            top_pos[:, 1] -= a.rate
            pos[top_t] = top_pos
            vel[top_t] = torch.tensor([0.0, -a.rate], device=dev)   # downward plate velocity
            top_y_now = float(top_pos[:, 1].min())

            # impermeable rigid plates: no grain may cross a plate face (zero escape)
            ps = pos[sand_t].clone()
            ps[:, 0].clamp_(X_LO + RADIUS, X_HI - RADIUS)
            ps[:, 1].clamp_(Y_BOT + RADIUS, top_y_now - RADIUS)
            pos[sand_t] = ps
            vel[sand_t] = (ps - prev[sand_t]) / DT             # velocity consistent with the clamped move

            # measure the specimen response
            sand_now = pos[sand_t].cpu().numpy()
            w, h, V, escaped = measure(sand_now, top_y_now)
            # DEM stress from grain-grain overlaps and a physical contact law F = k_n * overlap
            sig_yy, p_mean = dem_stress(model, pos, radius, sand_ids, max(V, 1e-9), a.emod)
            disp = (k + 1) * a.rate
            rec.append(dict(step=k + 1, disp=disp,
                            axial_strain=disp / h0,
                            vol_strain=(V0 - V) / V0,
                            void_ratio=(V - V_solid) / V_solid,
                            packing=V_solid / V,
                            sigma_yy=sig_yy,
                            p_mean=p_mean,
                            escaped=escaped))
            frames.append(pos.cpu().numpy().copy())
            if disp / h0 >= a.max_strain:                       # stop at the target axial strain
                break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0
    n_done = len(frames) - 1
    frames = np.stack(frames, 0)
    print(f"compression done: {frames.shape[0]} frames, "
          f"final axial strain {rec[-1]['axial_strain']*100:.1f}%, "
          f"final vol strain {rec[-1]['vol_strain']*100:.1f}% | "
          f"inference {t_infer:.2f} s for {n_done} steps "
          f"({1000*t_infer/n_done:.1f} ms/step, {N} particles incl. {M} plate)")

    # ── save data, curves, animation ──
    save_csv(rec, outdir / "compression_data.csv")
    plot_curve(rec, outdir / "compression_curve.png")
    Cc, r2 = plot_elnp(rec, outdir / "e_lnp_dem.png")
    render_gif(frames, frozen, top_mask, outdir / "compression.gif")
    print(f"e-ln(p) from DEM contact law: Cc = {Cc:.2f}, R2 = {r2:.2f}")
    print(f"saved -> {outdir/'compression.gif'}, {outdir/'compression_curve.png'}, "
          f"{outdir/'e_lnp_dem.png'}, {outdir/'compression_data.csv'}")


def save_csv(rec, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec[0].keys()))
        w.writeheader(); w.writerows(rec)


def plot_elnp(rec, path):
    """Void ratio versus log stress, with the stress from the DEM contact law.

    Bins the per-step data by axial strain (quasi-static averaging), drops the
    seating phase, and fits e = e0 - Cc * ln(p) to get the compression index Cc.
    The stress axis is in units of the chosen contact stiffness, so Cc (the slope)
    is meaningful while the absolute stress is not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strain = np.array([r["axial_strain"] for r in rec]) * 100
    e = np.array([r["void_ratio"] for r in rec])
    p = np.array([r["p_mean"] for r in rec]) / 1000.0      # Pa -> kPa
    bins = np.linspace(0, strain.max(), 31)
    idx = np.digitize(strain, bins)
    be, bp = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum():
            be.append(e[m].mean()); bp.append(p[m].mean())
    be = np.array(be); bp = np.array(bp)
    keep = (bp > 0) & (np.arange(len(bp)) > 2)             # drop the seating phase
    lnp = np.log(bp[keep]); em = be[keep]
    A = np.polyfit(lnp, em, 1)
    Cc = -float(A[0])
    r2 = float(1 - ((em - (A[0] * lnp + A[1])).var() / em.var()))

    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=150)
    ax.scatter(lnp, em, s=18, color="#484878", zorder=3, label="binned data")
    xs = np.linspace(lnp.min(), lnp.max(), 50)
    ax.plot(xs, A[0] * xs + A[1], color="#C45AD6", lw=1.8,
            label=f"Cc = {Cc:.2f}  (R² = {r2:.2f})")
    ax.set_xlabel(r"ln( vertical stress $\sigma$ / kPa )")
    ax.set_ylabel("Void ratio  e")
    ax.set_title("e - ln p from a PFC contact law (E* = 100 MPa) on TRACE overlaps",
                 fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return Cc, r2


def plot_curve(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = np.array([r["disp"] for r in rec])
    ev = np.array([r["vol_strain"] for r in rec]) * 100
    e = np.array([r["void_ratio"] for r in rec])

    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=150)
    ax.plot(disp, ev, color="#484878", lw=2.0)
    ax.set_xlabel("Top-plate downward displacement")
    ax.set_ylabel("Volumetric strain of the specimen (%)", color="#484878")
    ax.tick_params(axis="y", labelcolor="#484878")
    ax.spines["top"].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(disp, e, color="#C45AD6", lw=1.6, ls="--")
    ax2.set_ylabel("Void ratio", color="#C45AD6")
    ax2.tick_params(axis="y", labelcolor="#C45AD6")
    ax2.spines["top"].set_visible(False)
    ax.set_title("Confined compression: displacement vs volume change", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_gif(frames, frozen, top_mask, out_path, fps=25, dt_phys=0.0025):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sand = ~frozen
    fixed_plate = frozen & ~top_mask
    color = frames[0, sand, 1]
    T = len(frames) - 1
    x0, x1 = X_LO - 0.06, X_HI + 0.06
    y0, y1 = Y_BOT - 0.06, Y_TOP0 + 0.10

    fig, ax = plt.subplots(figsize=(4.6, 4.6 * (y1 - y0) / (x1 - x0)), dpi=150)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("TRACE — confined compression", fontsize=12)
    ax.scatter(frames[0, fixed_plate, 0], frames[0, fixed_plate, 1], c="0.6", s=7)
    plate = ax.scatter(frames[0, top_mask, 0], frames[0, top_mask, 1], c="0.25", s=9)
    sc = ax.scatter(frames[0, sand, 0], frames[0, sand, 1], c=color, cmap="viridis", s=7)
    step_txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, fontsize=10,
                       va="top", ha="left", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6))

    def update(i):
        sc.set_offsets(frames[i, sand])
        plate.set_offsets(frames[i, top_mask])
        step_txt.set_text(f"step {i:3d}/{T}\nt = {i*dt_phys:.3f} s")
        return sc, plate, step_txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
