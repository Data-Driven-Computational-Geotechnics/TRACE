"""统一 rollout 评估: 一个入口加载并自回归 trace / gns / tgnn 三类 checkpoint。

所有模型都在【帧 1】用真实初速 vel[1] 起步(与 render.rollout_pred 同协议), 用半隐式
Euler 前推 horizon 步, 各自模型自带边界处理, 返回逐帧对齐的 (pred_pos, gt_pos)。

- trace: 直接复用 render.rollout_pred (含粒子间投影 contact_projection, 属 Trace 方法一部分);
- gns:   节点速度历史窗口, 种子 vel_seq0 = vel[1] 复制 n_history 次(数据仅 vel[1] 为真);
- tgnn:  位置快照窗口, 种子 pos_seq0 = [pos0]*(L-1)+[pos1] (最新一帧速度 = 真实 vel[1])。

用法(库):
    from eval_any import load_any, rollout_any
    m, kind, cfg = load_any(ckpt, dev)
    pred, gt = rollout_any(m, kind, sample, dev, horizon, box)   # (R+1,N,d) numpy
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for _p in ("trace", "baselines/gns", "baselines/nmgns", "common"):
    _sp = str(REPO / _p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)


def _resolve_skin(cfg):
    """cfg 若已写回解析后的 skin_factor 直接用; 否则从数据集 metadata 推 R_conn/(2r)。"""
    sk = cfg.get("skin_factor")
    if sk is not None:
        return float(sk)
    meta = json.load(open(Path(cfg["dataset_root"]) / "metadata.json"))
    return float(meta["default_connectivity_radius"]) / (2.0 * float(cfg["radius"]))


def load_any(ckpt_path, dev):
    """按 config.model_name 分发, 重建模型并加载权重(归一化 buffer 随 state_dict 恢复)。
    返回 (model, kind, cfg)。"""
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ck["config"]
    kind = cfg.get("model_name", "trace")
    dim = int(cfg.get("dim", 2))
    skin = _resolve_skin(cfg)
    r = float(cfg["radius"]); box = float(cfg["box_size"])
    hid = int(cfg["hidden_dim"])

    if kind == "trace":
        if dim == 2:
            from tracegnn.model2d import Trace2D as Cls
        else:
            from tracegnn.model3d import Trace as Cls
        m = Cls(hidden_dim=hid, memory_dim=int(cfg["memory_dim"]),
                num_layers=int(cfg["num_layers"]), skin_factor=skin,
                memory_type=cfg.get("memory", "st"),
                normalize_inputs=bool(cfg.get("normalize_inputs", False)),
                boundary_features=bool(cfg.get("boundary_features", False))).to(dev)
        if cfg.get("boundary_features"):
            m.box_size.fill_(box); m.feat_clip.fill_(skin * 2 * r)
        m.vel_from_displacement = bool(cfg.get("vel_from_displacement", False))
    elif kind == "gns":
        from gns.model import GNS2D, GNS3D
        Cls = GNS2D if dim == 2 else GNS3D
        m = Cls(n_history=int(cfg["n_history"]), hidden_dim=hid,
                num_layers=int(cfg["num_layers"]), skin_factor=skin).to(dev)
    elif kind == "tgnn":
        from tgnn.model import TGNNS2D, TGNNS3D
        Cls = TGNNS2D if dim == 2 else TGNNS3D
        m = Cls(n_frames=int(cfg["n_frames"]), n_gnn=int(cfg["n_gnn"]),
                hidden_dim=hid, skin_factor=skin).to(dev)
    else:
        raise ValueError(f"未知 model_name: {kind!r}")

    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, kind, cfg


def rollout_any(m, kind, sample, dev, horizon, box):
    """返回 (pred_pos, gt_pos) 均为 (R+1, N, d) numpy, 帧 1 起对齐。"""
    if kind == "trace":
        import render as R
        return R.rollout_pred(m, sample, dev, 1.0, box, horizon)

    pos = torch.as_tensor(sample["pos"]).to(dev)
    vel = torch.as_tensor(sample["vel"]).to(dev)
    nt = torch.as_tensor(sample["node_type"]).to(dev)
    rad = torch.as_tensor(sample["radius"]).to(dev)
    bx = (0.0, float(box))
    with torch.no_grad():
        if kind == "gns":
            # GNS 从帧 1 起步(与 Trace 同锚点), 速度历史窗口用真实 vel[1] 填充
            H = int(m.n_history)
            Rn = min(horizon, pos.shape[0] - 2)
            vseq0 = vel[1].unsqueeze(1).repeat(1, H, 1)          # (N,H,d)
            traj = m.rollout(pos[1], vseq0, nt, rad, Rn, dt=1.0, box=bx)
            pred = torch.stack([pos[1]] + [t["pos"] for t in traj], 0)
            gt = pos[1: Rn + 2]
        elif kind == "tgnn":
            # TGNN 需要 L 帧历史: 用真实前 L 帧 pos[0:L] 预热(与其训练/验证一致),
            # 之后从帧 L-1 起自回归。伪造历史会污染节点记忆 -> 不公平。
            L = int(m.n_frames)
            Rn = min(horizon, pos.shape[0] - L)
            traj = m.rollout(pos[0:L], nt, rad, Rn, dt=1.0, box=bx)
            pred = torch.stack([pos[L - 1]] + [t["pos"] for t in traj], 0)
            gt = pos[L - 1: L - 1 + Rn + 1]
        else:
            raise ValueError(kind)
    return pred.cpu().numpy(), gt.cpu().numpy()
