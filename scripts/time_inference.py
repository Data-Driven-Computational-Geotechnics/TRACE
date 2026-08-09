#!/usr/bin/env python3
"""推理耗时(多次重复 -> mean±std)。公平: 同GPU/同轨迹/同horizon, warmup + cuda.synchronize。
每个模型跑 REPEATS 次完整 pass(每 pass 遍历全部轨迹), 报 ms/step 的均值±标准差(跨 pass = 测量抖动)。
推理速度与种子无关 -> 用 seed-0 checkpoint。须在空闲 GPU 上跑。

用法: python tools/time_inference.py <n_traj> <horizon> <gpu> <label>=<ckpt> [...]
      环境变量 TIMING_REPEATS 控制重复次数(默认 5)。
"""
import sys, json, time, os
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "common"))
from eval_any import load_any, rollout_any
from sand_data import SandH5Data

N_TRAJ = int(sys.argv[1]); HORIZON = int(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
PAIRS = [a.split("=", 1) for a in sys.argv[4:]]
REPEATS = int(os.environ.get("TIMING_REPEATS", "5"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP = 2

meta = json.load(open(REPO / "datasets/008-Sand-2D/metadata.json"))
bl = float(meta["bounds"][0][0])


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


rows = {}
for label, ckpt in PAIRS:
    m, kind, cfg = load_any(Path(ckpt), dev)
    box = float(cfg["box_size"]); r = float(cfg["radius"])
    data = SandH5Data(REPO / "datasets/008-Sand-2D/test.h5", n_samples=N_TRAJ,
                      radius=r, box_lower=bl, box_size=box).data
    Ntimed = data[WARMUP:]
    nparts = np.mean([np.asarray(s["pos"]).shape[1] for s in Ntimed])
    # 预热(不计时)
    for s in data[:WARMUP]:
        rollout_any(m, kind, s, dev, HORIZON, box)
    sync()
    # R 次完整 pass
    pass_msstep, pass_secroll = [], []
    for rep in range(REPEATS):
        tot_t, tot_steps, nroll = 0.0, 0, 0
        for s in Ntimed:
            sync(); t0 = time.perf_counter()
            pred, _ = rollout_any(m, kind, s, dev, HORIZON, box)
            sync(); dt = time.perf_counter() - t0
            tot_t += dt; tot_steps += pred.shape[0] - 1; nroll += 1
        pass_msstep.append(tot_t / tot_steps * 1000.0)     # ms/step
        pass_secroll.append(tot_t / nroll)                 # s/rollout
    pm = np.array(pass_msstep); ps = np.array(pass_secroll)
    rows[label] = dict(kind=kind, params=int(sum(p.numel() for p in m.parameters())),
                       repeats=REPEATS, mean_particles=float(nparts),
                       ms_per_step=float(pm.mean()), ms_per_step_std=float(pm.std(ddof=0)),
                       sec_per_rollout=float(ps.mean()), sec_per_rollout_std=float(ps.std(ddof=0)),
                       passes_msstep=[round(x, 3) for x in pass_msstep])
    print(f"[done] {label}: {pm.mean():.2f}±{pm.std(ddof=0):.2f} ms/step  (passes {rows[label]['passes_msstep']})", flush=True)

order = sorted(rows, key=lambda k: rows[k]["ms_per_step"])
print(f"\n=== 推理耗时 (n_timed={N_TRAJ-WARMUP}, horizon={HORIZON}, GPU={sys.argv[3]}, {REPEATS} 次重复) ===")
print(f"{'label':20s} {'kind':6s} {'粒子':>6s} {'ms/step (mean±std)':>20s} {'s/rollout':>14s}")
for k in order:
    r = rows[k]
    print(f"{k:20s} {r['kind']:6s} {r['mean_particles']:>6.0f} "
          f"{r['ms_per_step']:>10.2f} ± {r['ms_per_step_std']:<6.2f} "
          f"{r['sec_per_rollout']:>8.2f} ± {r['sec_per_rollout_std']:.2f}")

out = REPO / "results/_scores"; out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / f"timing_{HORIZON}.json", "w"), indent=1)
print(f"\n-> {out / f'timing_{HORIZON}.json'}")
