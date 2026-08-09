#!/usr/bin/env python3
"""Render a DEM-vs-model rollout comparison from a trained checkpoint.

Loads a ``best_model.pt`` (which carries its own config + normalizer buffers),
rolls the model out on one trajectory of a chosen split, and writes a static
multi-frame panel + an animated GIF under ``<run>/eval/rollout/`` (or --out).

Usage (from the repo root):
    python tools/render_rollout.py \
        --checkpoint experiments/01-2d-sand-collapse/01-trace/results/<exp>/<ts>/best_model.pt \
        --split test --traj 0 --n-steps 320

The model class is dispatched from the checkpoint's ``config.model_name`` so the
same driver serves trace / gns / tgnn once they have checkpoints.
"""
import sys, json, argparse
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "common"))
import render as R                                   # model/common/render.py
from sand_data import SandH5Data                     # model/common/sand_data.py


def build_model(cfg, dev):
    """Reconstruct the right model class from the checkpoint config."""
    name, dim = cfg["model_name"], int(cfg["dim"])
    if name == "trace":
        sys.path.insert(0, str(REPO / "trace"))
        if dim == 2:
            from tracegnn.model2d import Trace2D, Trace2D_NoMemory
            Cls = Trace2D_NoMemory if cfg.get("memory") == "none" else Trace2D
        else:
            from tracegnn.model3d import Trace, Trace_NoMemory
            Cls = Trace_NoMemory if cfg.get("memory") == "none" else Trace
        model = Cls(hidden_dim=int(cfg["hidden_dim"]), memory_dim=int(cfg["memory_dim"]),
                    num_layers=int(cfg["num_layers"]), skin_factor=float(cfg["skin_factor"]),
                    memory_type=cfg.get("memory", "st"),
                    normalize_inputs=bool(cfg.get("normalize_inputs", False)),
                    boundary_features=bool(cfg.get("boundary_features", False))).to(dev)
        # boundary scalars: set BEFORE load_state_dict so buffers exist & match;
        # rollout() re-pins box_size from its arg, so this is belt-and-suspenders.
        if bool(cfg.get("boundary_features", False)):
            model.box_size.fill_(float(cfg["box_size"]))
            model.feat_clip.fill_(float(cfg["skin_factor"]) * 2.0 * float(cfg["radius"]))
        # P3: 按训练时的协议恢复位移速度开关(旧 ckpt 无此键 → False, 行为不变)
        model.vel_from_displacement = bool(cfg.get("vel_from_displacement", False))
        return model
    raise NotImplementedError(f"render driver does not yet support model_name={name!r} "
                              f"(add its import in build_model)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to best_model.pt")
    ap.add_argument("--dataset-root", default=str(REPO / "datasets" / "008-Sand-2D"))
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--traj", type=int, default=0, help="trajectory index within the split")
    ap.add_argument("--n-steps", type=int, default=320, help="rollout horizon")
    ap.add_argument("--out", default=None, help="output dir (default <run>/eval/rollout)")
    ap.add_argument("--no-gif", action="store_true", help="skip the animated GIF")
    ap.add_argument("--gpu", default=None, help="CUDA device id, e.g. 0 (default: auto)")
    a = ap.parse_args()

    if a.gpu is not None:
        import os; os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(a.checkpoint)
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ck["config"]
    box_size, dt = float(cfg["box_size"]), float(cfg["dt"])

    model = build_model(cfg, dev)
    model.load_state_dict(ck["model_state_dict"])

    ds = Path(a.dataset_root)
    meta = json.load(open(ds / "metadata.json"))
    box_lower = float(meta["bounds"][0][0])
    data = SandH5Data(ds / f"{a.split}.h5", n_samples=a.traj + 1, radius=float(cfg["radius"]),
                      box_lower=box_lower, box_size=box_size).data
    sample = data[a.traj]

    out = Path(a.out) if a.out else ckpt_path.parent / "eval" / "rollout"
    name = f"{a.split}{a.traj:03d}"
    print(f"[render] model={cfg['model_name']} dim={cfg['dim']} | {a.split}#{a.traj} "
          f"N={sample['pos'].shape[1]} T={sample['pos'].shape[0]} | dev={dev} -> {out}")
    rmse = R.render_comparison(model, sample, dev, dt, box_size, a.n_steps, out,
                               name=name, make_gif=not a.no_gif)
    print(f"[render] done | RMSE @step{len(rmse)-1} = {rmse[-1]:.3e} | "
          f"panel={out/(name+'_panel.png')}")


if __name__ == "__main__":
    main()
