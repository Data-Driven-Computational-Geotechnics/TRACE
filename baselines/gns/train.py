#!/usr/bin/env python3
"""Trainer for the GNS baseline (this package) on a Sand dataset.

GNS carries temporal memory in NODE features as a short velocity history
(n_history frames). No edge memory, so no BPTT: each optimizer step accumulates
the normalized-acceleration Huber loss over `samples_per_traj` random time
indices. Best checkpoint by short-rollout position MSE.

Dataset-agnostic: spatial dimension comes from the dataset metadata (2D/3D).
    python model/02-gns/train.py --config gns/config/gns.yaml
Outputs land in <results_path>/<results_dirname>/results/<exp_name>/ (results_dirname
defaults to model_name; set it in the config, e.g. "02-gns"). See common/trainer_common.py.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # this package (gns)
sys.path.insert(0, str(HERE.parent.parent / "common"))     # trainer_common, sand_data
import trainer_common as tc


def vel_window(vel, t, H):
    """(N, H, dim) velocity history ending at frame t (oldest..newest)."""
    return vel[t - H + 1:t + 1].permute(1, 0, 2).contiguous()


def main():
    cfg = tc.load_config("gns")
    seed = int(cfg["seed"]); torch.manual_seed(seed)
    exp_name, ckpt_dir, log = tc.setup_run(cfg)
    dev = tc.get_device(cfg, log)

    train, val, info = tc.load_dataset(cfg)
    dim, radius, skin, box_size = info["dim"], info["radius"], info["skin"], info["box_size"]
    box = (0.0, box_size); dt = 1.0

    from gns.model import GNS2D, GNS3D
    GNSCls = GNS2D if dim == 2 else GNS3D

    H = int(cfg["n_history"]); epochs = int(cfg["epochs"])
    M = int(cfg.get("samples_per_traj", 6))
    val_rollout, noise, lr = int(cfg["val_rollout"]), float(cfg["noise"]), float(cfg["lr"])

    model = GNSCls(n_history=H, hidden_dim=int(cfg["hidden_dim"]),
                   num_layers=int(cfg["num_layers"]), skin_factor=skin).to(dev)
    model.set_accel_stats(*tc.fit_accel_stats(train, dt, dim))
    v_mean, v_std = tc.fit_vel_stats(train, dim)
    # vel_norm normalizes the flattened H-frame history -> tile per-dim stats across H
    model.set_vel_stats(v_mean.repeat(H), v_std.repeat(H))

    log.info("═" * 64)
    log.info(f"GNS (node velocity-history) · {dim}D sand · exp={exp_name}")
    log.info(f"config  | n_history={H}  hidden={cfg['hidden_dim']}  layers={cfg['num_layers']}  "
             f"samples/traj={M}  noise={noise:.1e}  lr={lr:.1e}  seed={seed}")
    log.info(f"data    | dim={dim}  N_train={len(train)}  N_val={len(val)}  radius={radius}  "
             f"skin={skin:.3f}  conn={2*radius*skin:.4f}  box={box_size:.3f}")
    log.info(f"model   | params={sum(p.numel() for p in model.parameters()):,}  "
             f"accel_std={[round(x,5) for x in model.accel_norm.std.tolist()]}")
    log.info(f"run     | epochs={epochs}  val_rollout={val_rollout}  ->  {ckpt_dir}")
    log.info("─" * 64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    total = epochs * len(train); warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / warm if st < warm else 0.5 * (1 + np.cos(np.pi * min(1.0, (st - warm) / max(1, total - warm)))))

    history = []; best = float("inf"); best_ep = -1; t0 = time.time()
    for ep in range(epochs):
        t_ep = time.time(); model.train(); tot = 0.0; nb = 0
        for si in torch.randperm(len(train)).tolist():
            s = train[si]
            pos = torch.as_tensor(s["pos"]).to(dev); vel = torch.as_tensor(s["vel"]).to(dev)
            nt = torch.as_tensor(s["node_type"]).to(dev); rad = torch.as_tensor(s["radius"]).to(dev)
            T = pos.shape[0]; opt.zero_grad(); loss = 0.0
            for _ in range(M):
                t = int(torch.randint(H - 1, T - 1, (1,)).item())
                vseq = vel_window(vel, t, H) + torch.randn(pos.shape[1], H, dim, device=dev) * noise
                pk = pos[t] + torch.randn_like(pos[t]) * noise
                tgt = model.accel_norm.normalize((vel[t + 1] - vel[t]) / dt)
                o = model(pk, vseq, nt, rad, box=box)
                loss = loss + torch.nn.functional.huber_loss(o["accel"], tgt)
            loss = loss / M; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
        trl = tot / max(nb, 1)

        model.eval(); vtot = 0.0; vn = 0
        with torch.no_grad():
            for s in val:
                pos = torch.as_tensor(s["pos"]).to(dev); vel = torch.as_tensor(s["vel"]).to(dev)
                nt = torch.as_tensor(s["node_type"]).to(dev); rad = torch.as_tensor(s["radius"]).to(dev)
                T = pos.shape[0]; R = min(val_rollout, T - 1 - H)
                vseq0 = vel[0:H].permute(1, 0, 2).contiguous()
                traj = model.rollout(pos[H - 1], vseq0, nt, rad, n_steps=R, dt=dt, box=box)
                pred = torch.stack([t["pos"] for t in traj], 0)
                vtot += torch.nn.functional.mse_loss(pred, pos[H:H + R]).item(); vn += 1
        vl = vtot / max(vn, 1)

        improved = vl < best
        if improved:
            best, best_ep = vl, ep
            torch.save({"model_state_dict": model.state_dict(),
                        "config": dict(model_name="gns", dim=dim, n_history=H, skin_factor=skin,
                                       radius=radius, box_size=box_size, dt=dt,
                                       hidden_dim=int(cfg["hidden_dim"]), num_layers=int(cfg["num_layers"]))},
                       ckpt_dir / "best_model.pt")
        cur_lr = sched.get_last_lr()[0]; sec = time.time() - t_ep
        history.append(dict(epoch=ep, train=trl, val=vl, lr=cur_lr, sec=round(sec, 2), best=improved))
        log.info(f"epoch {ep:3d}/{epochs} | train {trl:9.4f} | val {vl:.3e} | "
                 f"best {best:.3e} {'★' if improved else ' '} | lr {cur_lr:.2e} | {sec:5.1f}s")

    elapsed = (time.time() - t0) / 60.0
    tc.plot_curves(history, ckpt_dir, exp_name)
    log.info("─" * 64)
    log.info(f"DONE | best_val={best:.3e} @ epoch {best_ep} | elapsed {elapsed:.1f} min | "
             f"ckpt={ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
