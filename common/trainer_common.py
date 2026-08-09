"""Shared training utilities (model/common) for the three model packages:

  * trace  (model/01-trace)  — per-CONTACT edge memory          (the method)
  * gns    (model/02-gns)    — node velocity-history memory      (GNS baseline)
  * tgnn   (model/03-tgnn)   — node-level temporal state         (TGNNS baseline)

Each package has its OWN trainer (model/<pkg>/train.py) because their input
interfaces differ; everything they share lives here: config loading, GPU
selection, the run-folder layout, logging, normalizer fitting, metrics, and the
008-Sand data loader ([[sand_data]]). The trainers are dataset-agnostic — the
spatial dimension comes from the dataset metadata, so the same trainer serves
both 2D (008-Sand) and 3D (009-Sand-3D).

Config style: a FLAT YAML (keys grouped by `# section` comments), e.g.
    exp_name: "trace"        # run name -> output folder
    gpu: [0]                 # GPU id(s); [] / null -> inherit CUDA_VISIBLE_DEVICES
    model_name: "trace"      # which model package
    ... (dataset / geometry / model / training keys, all flat)

Run-folder layout:
    <results_path>/<model_name>/results/<exp_name>/<timestamp>/
        ├── train.log        timestamped log
        ├── best_model.pt    best weights
        └── eval/            evaluation outputs
              ├── eval.log
              └── figs/
"""
import os, sys, json, math, logging, argparse
from pathlib import Path
from datetime import datetime
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so sand_data is importable


def _set_visible_gpus(gpu):
    """Map the config's `gpu` (int | list | null) onto CUDA_VISIBLE_DEVICES.

    Respects an externally-set CUDA_VISIBLE_DEVICES (e.g. from run.sh), so the
    parallel sweep can still place each method on its own GPU without editing
    configs. Must run before the first CUDA call (it does — called from
    load_config, before the trainer touches torch.cuda)."""
    if gpu is None or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    ids = gpu if isinstance(gpu, (list, tuple)) else [gpu]
    if len(ids) == 0:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(g)) for g in ids)


def load_config(model_name: str) -> dict:
    """Parse --config (+ a few CLI overrides), select GPUs, return the config dict."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to the run's YAML config")
    ap.add_argument("--exp-name", default=None, help="override exp_name (the output folder)")
    ap.add_argument("--gpu", default=None, help="override gpu, e.g. '0' or '1,2,3'")
    ap.add_argument("--results-path", default=None, help="override results_path (default '.')")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg.setdefault("model_name", model_name)
    if a.exp_name is not None:      cfg["exp_name"] = a.exp_name
    if a.results_path is not None:  cfg["results_path"] = a.results_path
    if a.seed is not None:          cfg["seed"] = a.seed
    if a.gpu is not None:           cfg["gpu"] = [int(x) for x in a.gpu.split(",") if x != ""]
    cfg["_config_src"] = a.config
    _set_visible_gpus(cfg.get("gpu"))
    return cfg


def make_logger(log_file: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO); logger.handlers.clear(); logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def setup_run(cfg: dict):
    """Create the run folder, open the logger.

    Directory layout:
        <results_path>/<model_name>/results/<exp_name>/<timestamp>/
        └── train.log / best_model.pt
        └── eval/        (eval.log + figs/ prepared)

    Config is kept only in <method>/config/<method>.yaml (hand-edited template);
    no auto-snapshot is written to the results directory.

    Returns (exp_name, ckpt_dir, logger).
    """
    model_name = cfg["model_name"]
    exp_name = cfg.get("exp_name") or model_name
    results_path = Path(cfg.get("results_path", "."))
    # 输出目录名默认 = model_name; 项目文件夹带编号前缀时(如 01-trace)用 results_dirname 覆盖
    dirname = cfg.get("results_dirname", model_name)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = results_path / dirname / "results" / exp_name / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "eval" / "figs").mkdir(parents=True, exist_ok=True)
    logger = make_logger(ckpt_dir / "train.log", f"{model_name}.{exp_name}")
    return exp_name, ckpt_dir, logger


def make_scheduler(opt, cfg, steps_per_epoch):
    """Warmup -> cosine anneal to lr_min over `decay_epochs`, then HOLD at lr_min.

    Decouples LR annealing from the total epoch count. With the old schedule
    (cosine over ALL epochs), a long run held the peak LR for thousands of
    epochs and never fine-converged. Here the LR reaches lr_min by `decay_epochs`
    regardless of `epochs`, then stays at the floor for fine-tuning, so long runs
    actually converge (and early stopping ends them).

    Config keys (all optional, with sensible defaults):
      warmup_steps  : optimizer steps of linear warmup           (default 100)
      decay_epochs  : epochs over which LR anneals to lr_min      (default = epochs)
      lr_min        : LR floor held after annealing               (default lr/100)
    """
    base_lr = float(cfg["lr"])
    warmup = int(cfg.get("warmup_steps", 100))
    decay_epochs = int(cfg.get("decay_epochs") or cfg["epochs"])
    lr_min = float(cfg.get("lr_min", base_lr * 0.01))
    floor = max(0.0, min(1.0, lr_min / base_lr)) if base_lr > 0 else 0.0
    horizon = max(1, decay_epochs * steps_per_epoch - warmup)

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / horizon)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def get_device(cfg, log=None):
    """Resolve the training device (cuda:0 within the CUDA_VISIBLE_DEVICES view)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log is not None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
        n = torch.cuda.device_count() if dev.type == "cuda" else 0
        log.info(f"device  | {dev.type}  gpu(config)={cfg.get('gpu')}  visible={vis}  count={n}"
                 + ("  [training uses the first visible GPU]" if n > 1 else ""))
    return dev


def load_dataset(cfg: dict):
    """Load the Sand train/val split and resolve geometry (radius, skin, box, dim).

    Reads `dim` and bounds from the dataset metadata, so the same trainer works
    for 2D (008-Sand) and 3D (009-Sand-3D). Returns (train_list, val_list, info)
    where info has dim/radius/skin/box_size/R_conn.
    """
    from sand_data import SandH5Data
    ds = Path(cfg["dataset_root"]); meta = json.load(open(ds / "metadata.json"))
    dim = int(meta["dim"])
    R_conn = float(meta["default_connectivity_radius"])
    box_lower, box_hi = float(meta["bounds"][0][0]), float(meta["bounds"][0][1])
    box_size = box_hi - box_lower
    radius = float(cfg["radius"])
    skin = float(cfg["skin_factor"]) if cfg.get("skin_factor") is not None else R_conn / (2.0 * radius)
    tr = SandH5Data(ds / "train.h5", n_samples=int(cfg["n_train"]), radius=radius,
                    box_lower=box_lower, box_size=box_size)
    va = SandH5Data(ds / "valid.h5", n_samples=int(cfg["n_val"]), radius=radius,
                    box_lower=box_lower, box_size=box_size)
    return tr.data, va.data, dict(dim=dim, radius=radius, skin=skin, box_size=box_size, R_conn=R_conn)


def fit_accel_stats(data, dt, dim):
    """Per-channel mean/std of the second difference a[t] = (v[t+1]-v[t])/dt."""
    accs = [(torch.as_tensor(s["vel"])[1:] - torch.as_tensor(s["vel"])[:-1]).reshape(-1, dim) / dt
            for s in data]
    accs = torch.cat(accs, 0)
    return accs.mean(0), accs.std(0)


def fit_vel_stats(data, dim):
    """Per-channel mean/std of the velocity v[t] = pos[t]-pos[t-1]."""
    vs = [torch.as_tensor(s["vel"]).reshape(-1, dim) for s in data]
    vs = torch.cat(vs, 0)
    return vs.mean(0), vs.std(0)


@torch.no_grad()
def fit_edge_stats(model, data, dim, noise, n_frames=512, device="cpu"):
    """Zero-centered per-channel std of the edge feature vector [rel_pos, rel_vel,
    dist] (width 2*dim+1), for the scale-only ``edge_norm``.

    Two things make this non-trivial (see design spec D4):
      * the std must be measured ABOUT ZERO (sqrt(mean(x**2))) because edge_norm
        is scale-only (mean forced to 0) to keep the reverse-edge antisymmetry;
      * it MUST be fit on the SAME GNS-noised distribution the model sees at
        train time. The clean ``rel_vel`` std is ~6e-8; fitting clean and then
        dividing the noised input (rel_vel std ~sqrt(2)*noise ~1e-4) would
        explode that channel by ~1000x. So we add the configured noise here too.

    Graphs are built via ``model._build_graph`` (with normalize_inputs forced off
    so it returns RAW edge_attr), so connectivity matches training exactly.
    """
    cols = []
    seen = 0
    prev = model.normalize_inputs
    model.normalize_inputs = False                 # force RAW edge_attr out of _build_graph
    try:
        for s in data:
            pos = torch.as_tensor(s["pos"]); vel = torch.as_tensor(s["vel"])
            nt = torch.as_tensor(s["node_type"]).to(device)
            rad = torch.as_tensor(s["radius"]).to(device)
            T = pos.shape[0]
            for t in range(1, T):                  # skip t=0 (vel[0]=0)
                p = (pos[t] + torch.randn_like(pos[t]) * noise).to(device)
                v = (vel[t] + torch.randn_like(vel[t]) * noise).to(device)
                _, _, edge_attr, *_ = model._build_graph(p, v, nt, rad)
                if edge_attr.shape[0] > 0:
                    cols.append(edge_attr.detach().cpu())
                seen += 1
                if seen >= n_frames:
                    break
            if seen >= n_frames:
                break
    finally:
        model.normalize_inputs = prev
    if not cols:
        return torch.ones(2 * dim + 1)             # degenerate: no edges -> unit std
    flat = torch.cat(cols, 0)                      # (M, 2*dim+1)
    return flat.pow(2).mean(0).sqrt()              # zero-centered std


def plot_curves(history, ckpt_dir, exp_name):
    """Save training curves (loss + LR) to <ckpt_dir>/training_curves.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [h["epoch"] for h in history]
    # 统一指标：train/val 都是 rollout 位置 RMSE(域单位); 兼容旧 history 的 train/val 键
    train_loss = [h.get("train_rmse", h.get("train")) for h in history]
    val_loss = [h.get("val_rmse", h.get("val")) for h in history]
    lrs = [h["lr"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(eps, train_loss, color="#2c3e50", lw=1.2, label="train RMSE (rollout)")
    ax1.plot(eps, val_loss, color="#c0392b", lw=1.5, label="val RMSE (rollout)")
    best_ep = min(enumerate(val_loss), key=lambda x: x[1])[0]
    ax1.axvline(eps[best_ep], color="#c0392b", ls="--", lw=0.8, alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Rollout position RMSE (log scale)")
    ax1.legend(loc="upper right")
    ax1.set_title(f"TRACE — {exp_name}  |  best val-RMSE={min(val_loss):.3e} @ epoch {eps[best_ep]}",
                  fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps, lrs, color="#27ae60", lw=1.2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate (log scale)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(ckpt_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

