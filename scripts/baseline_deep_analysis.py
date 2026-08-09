#!/usr/bin/env python3
"""基线对比的深度维度: 逐步 RMSE(t) 增长曲线 / 逐轨迹 RMSE 分布 / 物理可容许性(颗粒穿透).
用法: python tools/baseline_deep_analysis.py <n_traj> <horizon> <gpu>
输出 JSON: results/_scores/baseline_deep.json (供绘图)."""
import sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "common"))
from eval_any import load_any, rollout_any
from sand_data import SandH5Data

N_TRAJ = int(sys.argv[1]); HORIZON = int(sys.argv[2])
import os; os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta = json.load(open(REPO / "datasets/008-Sand-2D/metadata.json"))
bl = float(meta["bounds"][0][0])

CKPTS = {
    "Trace": REPO/"checkpoints/sand2d/trace_2d_final.pt",
    "GNS":   REPO/"checkpoints/sand2d/gns_2d.pt",
    "NMGNS": REPO/"checkpoints/sand2d/nmgns_2d.pt",
}

def penetration(pos, r, dev):
    """每帧平均颗粒穿透深度(占直径比例, 只算重叠对). pos:(N,2) numpy. 接触直径=2r."""
    twor = 2.0 * r
    p = torch.as_tensor(pos, dtype=torch.float32, device=dev)
    N = p.shape[0]
    with torch.no_grad():
        dmat = torch.cdist(p, p)                              # (N,N)
        iu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=dev), diagonal=1)
        m = iu & (dmat < twor)                                # 上三角 + 重叠
        if not m.any(): return 0.0
        overlap = (twor - dmat[m]) / twor
        return float(overlap.mean().item())

out = {}
for name, ck in CKPTS.items():
    m, kind, cfg = load_any(Path(ck), dev)
    box = float(cfg["box_size"]); r = float(cfg["radius"])
    data = SandH5Data(REPO/"datasets/008-Sand-2D/test.h5", n_samples=N_TRAJ,
                      radius=r, box_lower=bl, box_size=box).data
    per_traj_roll = []; step_rmse_acc = None; step_pen_acc = None; ntraj = 0
    for s in data:
        pred, gt = rollout_any(m, kind, s, dev, HORIZON, box)   # (T,N,2)
        e = np.sqrt(((pred - gt)**2).sum(-1).mean(-1))          # (T,) 逐帧 RMSE
        per_traj_roll.append(float(e.mean()))
        pen = np.array([penetration(pred[t], r, dev) for t in range(pred.shape[0])])  # (T,)
        L = e.shape[0]
        if step_rmse_acc is None:
            step_rmse_acc = np.zeros(L); step_pen_acc = np.zeros(L); cnt = np.zeros(L)
        Lc = min(L, step_rmse_acc.shape[0])
        step_rmse_acc[:Lc] += e[:Lc]; step_pen_acc[:Lc] += pen[:Lc]; ntraj += 1
    out[name] = dict(
        kind=kind, r=r,
        per_traj_roll=per_traj_roll,
        step_rmse=(step_rmse_acc/ntraj).tolist(),
        step_pen=(step_pen_acc/ntraj).tolist(),
        mean_pen=float(np.mean(step_pen_acc/ntraj)),
    )
    print(f"[done] {name}: roll={np.mean(per_traj_roll):.4f}  mean_penetration={out[name]['mean_pen']*100:.3f}% dia", flush=True)

od = REPO/"results/_scores"; od.mkdir(exist_ok=True)
json.dump(out, open(od/"baseline_deep.json","w"))
print("-> baseline_deep.json")
