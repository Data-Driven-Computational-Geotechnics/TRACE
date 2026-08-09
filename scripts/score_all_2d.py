#!/usr/bin/env python3
"""统一评估一批 checkpoint(trace/gns/tgnn 混合), 长时 rollout, 出对比/消融数字。

对每个 checkpoint 在 N 条 2D 测试轨迹上做 horizon 步自回归, 统计:
  roll_avg  = 逐帧位置 RMSE 的时间平均 (rollout-averaged)
  final     = 末帧位置 RMSE
  deposit   = (|Δheight| + |Δrunout|)/box 的均值 (沉积几何误差)
  params    = 参数量
所有模型经 eval_any 统一加载/前推(帧 1 种子), 保证协议一致、可比。

用法: python tools/score_all_2d.py <n_traj> <horizon> <gpu> <label>=<ckpt> [<label>=<ckpt> ...]
输出: 打印对齐表 + 写 JSON 到 results/_scores/scores_<horizon>.json
"""
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
PAIRS = [a.split("=", 1) for a in sys.argv[4:]]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta = json.load(open(REPO / "datasets/008-Sand-2D/metadata.json"))
bl = float(meta["bounds"][0][0])


def runout(p):
    return p[:, 0].max() - p[:, 0].min()


rows = {}
for label, ckpt in PAIRS:
    m, kind, cfg = load_any(Path(ckpt), dev)
    box = float(cfg["box_size"]); r = float(cfg["radius"])
    data = SandH5Data(REPO / "datasets/008-Sand-2D/test.h5", n_samples=N_TRAJ,
                      radius=r, box_lower=bl, box_size=box).data
    roll, final, dep = [], [], []
    for s in data:
        pred, gt = rollout_any(m, kind, s, dev, HORIZON, box)
        e = np.sqrt(((pred - gt) ** 2).sum(-1).mean(-1))          # (T,) 逐帧 RMSE
        roll.append(float(e.mean())); final.append(float(e[-1]))
        hp, hg = pred[-1, :, 1].max(), gt[-1, :, 1].max()
        dep.append(float((abs(hp - hg) + abs(runout(pred[-1]) - runout(gt[-1]))) / box))
    rows[label] = dict(kind=kind, params=int(sum(p.numel() for p in m.parameters())),
                       roll_avg=float(np.mean(roll)), roll_std=float(np.std(roll)),
                       final=float(np.mean(final)), deposit=float(np.mean(dep)), n=N_TRAJ)
    print(f"[done] {label}", flush=True)

order = sorted(rows, key=lambda k: rows[k]["roll_avg"])
print(f"\n=== 2D 统一评估 (n_traj={N_TRAJ}, horizon={HORIZON}, 帧1种子长时自回归) ===")
print(f"{'label':22s} {'kind':6s} {'params':>10s} {'roll_avg':>10s} {'final':>8s} {'deposit':>8s}")
for k in order:
    rr = rows[k]
    print(f"{k:22s} {rr['kind']:6s} {rr['params']:>10,} "
          f"{rr['roll_avg']:>10.4f} {rr['final']:>8.4f} {rr['deposit']:>8.4f}")

out = REPO / "results/_scores"
out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / f"scores_{HORIZON}.json", "w"), indent=1)
print(f"\n-> {out / f'scores_{HORIZON}.json'}")
