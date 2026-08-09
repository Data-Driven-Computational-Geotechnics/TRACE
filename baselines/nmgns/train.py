#!/usr/bin/env python3
"""Trainer for the TGNNS baseline (this package) on a Sand dataset.

TGNNS consumes a sequence of `n_frames` position snapshots and carries a
NODE-level temporal state between per-frame graphs. No BPTT across optimizer
steps: each step accumulates the normalized-acceleration Huber loss over
`samples_per_traj` random windows. Best checkpoint by short-rollout position MSE.

Dataset-agnostic: spatial dimension comes from the dataset metadata (2D/3D).
    python model/03-tgnn/train.py --config tgnn/config/tgnn.yaml
Outputs land in <results_path>/<results_dirname>/results/<exp_name>/ (results_dirname
defaults to model_name; set it in the config, e.g. "03-tgnn"). See common/trainer_common.py.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # this package (tgnn)
sys.path.insert(0, str(HERE.parent.parent / "common"))     # trainer_common, sand_data
import trainer_common as tc


def main():
    cfg = tc.load_config("tgnn")
    seed = int(cfg["seed"]); torch.manual_seed(seed)
    exp_name, ckpt_dir, log = tc.setup_run(cfg)
    dev = tc.get_device(cfg, log)

    train, val, info = tc.load_dataset(cfg)
    dim, radius, skin, box_size = info["dim"], info["radius"], info["skin"], info["box_size"]
    box = (0.0, box_size); dt = 1.0

    from tgnn.model import TGNNS2D, TGNNS3D
    TGNNSCls = TGNNS2D if dim == 2 else TGNNS3D

    L = int(cfg["n_frames"]); epochs = int(cfg["epochs"])
    M = int(cfg.get("samples_per_traj", 6))
    val_rollout, noise, lr = int(cfg["val_rollout"]), float(cfg["noise"]), float(cfg["lr"])

    model = TGNNSCls(n_frames=L, n_gnn=int(cfg["n_gnn"]),
                     hidden_dim=int(cfg["hidden_dim"]), skin_factor=skin).to(dev)
    model.set_accel_stats(*tc.fit_accel_stats(train, dt, dim))
    model.set_vel_stats(*tc.fit_vel_stats(train, dim))

    log.info("═" * 64)
    log.info(f"TGNNS (node temporal state) · {dim}D sand · exp={exp_name}")
    log.info(f"config  | n_frames={L}  n_gnn={cfg['n_gnn']}  hidden={cfg['hidden_dim']}  "
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
                t = int(torch.randint(L - 1, T - 1, (1,)).item())
                pseq = pos[t - L + 1:t + 1] + torch.randn(L, pos.shape[1], dim, device=dev) * noise
                tgt = model.accel_norm.normalize((vel[t + 1] - vel[t]) / dt)
                o = model(pseq, nt, rad, dt=dt, box=box)
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
                T = pos.shape[0]; R = min(val_rollout, T - 1 - L)
                traj = model.rollout(pos[0:L], nt, rad, n_steps=R, dt=dt, box=box)
                pred = torch.stack([t["pos"] for t in traj], 0)
                vtot += torch.nn.functional.mse_loss(pred, pos[L:L + R]).item(); vn += 1
        vl = vtot / max(vn, 1)

        improved = vl < best
        if improved:
            best, best_ep = vl, ep
            torch.save({"model_state_dict": model.state_dict(),
                        "config": dict(model_name="tgnn", dim=dim, n_frames=L, n_gnn=int(cfg["n_gnn"]),
                                       skin_factor=skin, radius=radius, box_size=box_size, dt=dt,
                                       hidden_dim=int(cfg["hidden_dim"]))},
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
