"""
PhysicsNeMo-style Distributed Training Shell for Trace.

This module provides a lightweight, self-contained distributed training
infrastructure that follows PhysicsNeMo's API patterns:

    from tracegnn.distributed import DistributedManager, PhysicsNeMoTrainer

    dist = DistributedManager.initialize()
    trainer = PhysicsNeMoTrainer(model, config, dist)
    trainer.train()

When you're ready to use the real PhysicsNeMo, replace:

    from tracegnn.distributed import DistributedManager
    →  from physicsnemo.distributed import DistributedManager

    from tracegnn.distributed import PhysicsNeMoTrainer
    →  write a thin wrapper around physicsnemo.launch

The API signatures are intentionally compatible.

Requirements:
    - torch (with distributed support for multi-GPU)
    - torch_geometric
    - wandb (optional, for logging)
    - omegaconf (for Hydra-style config)
"""

import os
import sys
import math
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trace")


# ═══════════════════════════════════════════════════════════════════
# DistributedManager — PhysicsNeMo-compatible API
# ═══════════════════════════════════════════════════════════════════

class DistributedManager:
    """
    PhysicsNeMo-compatible distributed environment manager.

    Usage (PhysicsNeMo-style, identical API):
        dist_manager = DistributedManager.initialize()
        print(f"World size: {dist_manager.world_size}")
        print(f"Rank: {dist_manager.rank}")
        model = model.to(dist_manager.device)
        if dist_manager.distributed:
            model = DDP(model, device_ids=[dist_manager.local_rank])
    """

    _instance: Optional["DistributedManager"] = None

    def __init__(self):
        self._distributed = False
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._master = True
        # MUST be False: the frozen accel-normalizer buffers (mean/std) are fit
        # identically on every rank, so they need no sync; broadcasting them
        # in-place each forward would corrupt the multi-step BPTT graph
        # ("variable modified by an inplace operation").
        self._broadcast_buffers = False
        self._find_unused_parameters = False
        self._initialized = False

    @classmethod
    def initialize(
        cls,
        backend: str = "nccl",
        init_method: str = "env://",
        port: Optional[int] = None,
    ) -> "DistributedManager":
        """
        Initialize the distributed environment.

        In production (PhysicsNeMo), this wraps torch.distributed.init_process_group.
        For single-GPU or CPU, it's a no-op singleton.
        """
        if cls._instance is not None:
            return cls._instance

        instance = cls()
        cls._instance = instance

        # Check if launched via torchrun / mpirun
        env_rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        env_world = os.environ.get("WORLD_SIZE")

        if env_rank is not None and env_world is not None:
            instance._distributed = True
            instance._rank = int(os.environ.get("RANK", 0))
            instance._world_size = int(os.environ.get("WORLD_SIZE", 1))
            instance._local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                )

            torch.cuda.set_device(instance._local_rank)
            instance._device = torch.device(f"cuda:{instance._local_rank}")
            instance._master = (instance._rank == 0)

            logger.info(
                f"[DistributedManager] Initialized: "
                f"rank={instance._rank}/{instance._world_size}, "
                f"local_rank={instance._local_rank}, "
                f"device={instance._device}"
            )
        else:
            # Single-GPU / CPU mode
            if torch.cuda.is_available():
                instance._device = torch.device("cuda:0")
            instance._distributed = False
            instance._rank = 0
            instance._world_size = 1
            instance._local_rank = 0
            instance._master = True
            logger.info(
                f"[DistributedManager] Single-device mode: {instance._device}"
            )

        instance._initialized = True
        return instance

    @classmethod
    def cleanup(cls):
        """Clean up the distributed environment."""
        if cls._instance is not None and cls._instance._distributed:
            if dist.is_initialized():
                dist.destroy_process_group()
        cls._instance = None

    # ── Properties (PhysicsNeMo-compatible names) ──

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def distributed(self) -> bool:
        return self._distributed

    @property
    def broadcast_buffers(self) -> bool:
        return self._broadcast_buffers

    @property
    def find_unused_parameters(self) -> bool:
        return self._find_unused_parameters

    @property
    def is_master(self) -> bool:
        return self._master

    # ── Utility methods ──

    def barrier(self):
        """Synchronize all processes."""
        if self._distributed:
            dist.barrier()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src to all ranks."""
        if self._distributed:
            dist.broadcast(tensor, src=src)
        return tensor

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce across all ranks."""
        if self._distributed:
            dist.all_reduce(tensor, op=op)
        return tensor

    def print(self, *args, **kwargs):
        """Print only on master rank."""
        if self._master:
            print(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Training configuration (Hydra/OmegaConf compatible)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TraceConfig:
    """Trace training configuration (PhysicsNeMo Hydra-compatible)."""

    # ── Model ──
    model: Dict[str, Any] = field(default_factory=lambda: {
        "node_in_dim": 8,
        "edge_in_dim": 7,   # rel_pos(3) + rel_vel(3) + dist(1)
        "hidden_dim": 128,
        "memory_dim": 16,
        "num_layers": 10,
        "num_history": 4,
        "output_accel": True,
        "noise_std": 3e-4,        # input-state noise during training (physical units)
        "connect_radius": 0.05,   # contact-graph radius (~2x particle diameter)
    })

    # ── Training ──
    training: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 1,
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "warmup_steps": 300,      # step-based warmup (replaces epoch-based)
        "rollout_steps": 1,       # 1 = single-step; >1 = truncated BPTT over edge memory
        "val_rollout_steps": 15,  # autoregressive steps for the validation metric
        "loss_type": "mse",       # "mse" | "huber"
        "position_weight": 1.0,
        "velocity_weight": 1.0,
    })

    # ── Data ──
    data: Dict[str, Any] = field(default_factory=lambda: {
        "data_dir": "./data/dem",
        "use_synthetic": True,    # True for quick test, False for real DEM
        "n_particles": 400,
        "n_steps": 150,
        "n_train": 200,
        "n_val": 20,
        "n_test": 20,
        "dt": 0.003,              # RECORDED timestep (= dt_phys * substeps)
        "substeps": 20,           # physics sub-steps per recorded frame
        "cache_dir": "/data/tmp_cmgn/dem_cache",
    })

    # ── Distributed ──
    distributed: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "nccl",
        "find_unused_parameters": False,
    })

    # ── Logging ──
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "log_dir": "./logs/cmgn",
        "wandb_mode": "disabled",   # "online" | "offline" | "disabled"
        "wandb_project": "cmgn",
        "wandb_entity": None,
        "log_interval": 10,         # steps
        "save_interval": 10,        # epochs
    })

    # ── Checkpoint ──
    checkpoint: Dict[str, Any] = field(default_factory=lambda: {
        "save_dir": "./checkpoints/cmgn",
        "resume_dir": None,         # set to resume training
        "save_best": True,
    })


# ═══════════════════════════════════════════════════════════════════
# PhysicsNeMo-style Trainer
# ═══════════════════════════════════════════════════════════════════

class PhysicsNeMoTrainer:
    """
    PhysicsNeMo-compatible training loop for Trace.

    Provides the same training infrastructure as PhysicsNeMo's launch module:
    - Multi-GPU DDP
    - WandB / MLFlow logging
    - Checkpoint save/resume
    - Hydra config integration
    - Learning rate scheduling

    Usage:
        config = TraceConfig()
        model = Trace(**config.model)
        trainer = PhysicsNeMoTrainer(model, config, dist_manager)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TraceConfig,
        dist_manager: Optional[DistributedManager] = None,
    ):
        self.dist = dist_manager or DistributedManager.initialize()
        self.config = config
        self.device = self.dist.device
        self.is_master = self.dist.is_master

        # ── Move model to device ──
        self.model = model.to(self.device)

        # ── Wrap with DDP if distributed ──
        if self.dist.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # ── Access underlying model (unwrapped from DDP) ──
        self._raw_model = self.model.module if self.dist.distributed else self.model

        # ── Optimizer & Scheduler ──
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training["lr"],
            weight_decay=config.training["weight_decay"],
        )

        self.scheduler = self._build_scheduler()

        # ── Metrics tracking ──
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.metrics_history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "lr": []
        }

        # ── Logging setup ──
        self._setup_logging()
        self._setup_wandb()

        # ── Create directories ──
        if self.is_master:
            Path(config.logging["log_dir"]).mkdir(parents=True, exist_ok=True)
            Path(config.checkpoint["save_dir"]).mkdir(parents=True, exist_ok=True)

    def _build_scheduler(self):
        """Step-based LR schedule: linear warmup then cosine decay.

        Stepped once per OPTIMIZER step (not per epoch). With batch_size=1 and
        per-sample steps, steps_per_epoch ~= n_train, so the cosine is smooth
        instead of advancing only once per epoch (the old epoch-based bug that
        left short runs entirely in warmup at LR~0).
        """
        cfg = self.config.training
        warmup = max(1, int(cfg.get("warmup_steps", 300)))
        # Optimizer steps per epoch = samples PER RANK (DistributedSampler shards
        # the train set across ranks), so the cosine spans the real step count.
        world = max(1, self.dist.world_size)
        steps_per_epoch = max(1, math.ceil(int(self.config.data.get("n_train", 200)) / world))
        total_steps = max(warmup + 1, cfg["epochs"] * steps_per_epoch)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_logging(self):
        """Set up file-based logging."""
        if not self.is_master:
            return
        log_dir = Path(self.config.logging["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)
        logger.info(f"Training log: {self.log_file}")

    def _setup_wandb(self):
        """Set up Weights & Biases logging (optional)."""
        cfg = self.config.logging
        self.use_wandb = False

        if cfg.get("wandb_mode", "disabled") == "disabled":
            return

        try:
            import wandb
            if self.is_master:
                wandb.init(
                    project=cfg.get("wandb_project", "cmgn"),
                    entity=cfg.get("wandb_entity"),
                    config=self.config.__dict__,
                    mode=cfg["wandb_mode"],
                    dir=cfg["log_dir"],
                )
                self.use_wandb = True
                logger.info("WandB logging enabled")
        except ImportError:
            logger.warning("wandb not installed — skipping WandB logging")

    def _compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        For EMERGENT memory (Trace headline): loss is only on macroscopic
        observables (position, velocity). Per-contact forces are NOT supervised.
        """
        cfg = self.config.training
        loss_type = cfg.get("loss_type", "mse")

        if loss_type == "huber":
            loss_fn = lambda p, t: torch.nn.functional.huber_loss(p, t, delta=1.0)
        else:
            loss_fn = torch.nn.functional.mse_loss

        # Loss in NORMALIZED acceleration space (pred["accel"] is already
        # normalized; normalize the target the same way). This is the fix that
        # lets the loss actually descend instead of plateauing on the
        # gravity-dominated, un-normalized target.
        tgt = self._raw_model.accel_norm.normalize(target["accel"])
        accel_loss = loss_fn(pred["accel"], tgt)

        total = accel_loss
        losses = {"total": total, "position": accel_loss,
                  "velocity": torch.zeros((), device=self.device)}

        # Supervised variant: optional per-contact tangential-force loss.
        if hasattr(self._raw_model, "tangent_loss_weight"):
            if "F_t_true" in target and "F_t" in pred:
                tangent_loss = self._raw_model.tangent_loss(pred["F_t"], target["F_t_true"])
                total = total + self._raw_model.tangent_loss_weight * tangent_loss
                losses["tangent"] = tangent_loss
                losses["total"] = total

        return losses

    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train one epoch with K-step truncated BPTT over the edge memory.

        For each sample we pick a start t0 and unroll K steps, CARRYING the edge
        memory across steps (teacher-forced inputs from ground truth, so targets
        stay consistent). The memory is kept attached across the K steps so the
        GRU receives cross-step gradient — the whole point of Trace. Inputs get
        small GNS-style noise. One optimizer step per unrolled window.
        """
        self.model.train()
        epoch_losses = {"total": 0.0, "position": 0.0, "velocity": 0.0}
        n_samples = 0
        last_grad = 0.0

        K = int(self.config.training.get("rollout_steps", 1))
        dt = self.config.data["dt"]
        noise = float(getattr(self._raw_model, "noise_std", 0.0) or 0.0)
        grad_clip = self.config.training.get("grad_clip", 1.0)

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["pos_seq"].shape[0]
            T = batch["pos_seq"].shape[1]

            for b in range(B):
                Kk = max(1, min(K, T - 1))
                hi = T - 1 - Kk
                t0 = 0 if hi <= 0 else torch.randint(0, hi + 1, (1,)).item()
                node_type = batch["node_type"][b]
                radius = batch["radius"][b]

                self.optimizer.zero_grad()
                memory, id_map = None, None
                total_loss = 0.0
                for k in range(Kk):
                    pos_k = batch["pos_seq"][b, t0 + k]
                    vel_k = batch["vel_seq"][b, t0 + k]
                    if noise > 0:
                        pos_k = pos_k + torch.randn_like(pos_k) * noise
                        vel_k = vel_k + torch.randn_like(vel_k) * noise
                    target_accel = (
                        batch["vel_seq"][b, t0 + k + 1] - batch["vel_seq"][b, t0 + k]
                    ) / dt
                    # Use the RAW (unwrapped) model: DDP cannot handle multiple
                    # forwards before a single backward (the K-step BPTT). We
                    # all-reduce gradients manually below instead.
                    pred = self._raw_model(
                        pos=pos_k, vel=vel_k, node_type=node_type, radius=radius,
                        material_id=batch.get("material_id", None),
                        edge_memory_state=memory, edge_id_map=id_map,
                    )
                    losses = self._compute_loss(pred, {"accel": target_accel})
                    total_loss = total_loss + losses["total"]
                    memory = pred["edge_memory_state"]   # attached -> BPTT
                    id_map = pred["edge_id_map"]

                total_loss = total_loss / Kk
                total_loss.backward()

                # Manual gradient all-reduce across ranks (DDP's autograd hooks
                # are bypassed since we used the raw model for the unroll).
                if self.dist.distributed:
                    ws = self.dist.world_size
                    for p in self._raw_model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(ws)

                last_grad = torch.nn.utils.clip_grad_norm_(
                    self._raw_model.parameters(), grad_clip
                ).item()
                self.optimizer.step()
                self.scheduler.step()   # step-based LR

                epoch_losses["total"] += total_loss.item()
                epoch_losses["position"] += total_loss.item()
                n_samples += 1

            if self.is_master and batch_idx % self.config.logging["log_interval"] == 0:
                running = epoch_losses["total"] / max(n_samples, 1)
                logger.info(
                    f"Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                    f"Loss(run) {running:.6f} | grad {last_grad:.2e} | "
                    f"LR {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": running,
                        "train/grad_norm": last_grad,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "step": self.global_step,
                    })

            self.global_step += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_samples, 1)
        return epoch_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation = short autoregressive ROLLOUT position error.

        This is the metric that's actually comparable across configs (unlike the
        old t=0->1 one-step accel MSE, which only measured the near-static
        gravity regime and made val look deceptively good).
        """
        self.model.eval()
        dt = self.config.data["dt"]
        R_cfg = int(self.config.training.get("val_rollout_steps", 15))
        total = 0.0
        n_samples = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["pos_seq"].shape[0]
            T = batch["pos_seq"].shape[1]
            R = max(1, min(R_cfg, T - 1))
            # NOTE: this legacy distributed path calls rollout WITHOUT box_size (defaults to 1.0)
            # and does not fit input-normalizer stats. Until it is wired up, fail loudly rather
            # than silently mis-scale boundary features / skip input normalization (use train.py).
            assert not getattr(self._raw_model, "boundary_features", False), \
                "distributed.py does not support boundary_features yet (no box_size plumbed); use train.py"
            for b in range(B):
                traj = self._raw_model.rollout(
                    batch["pos_seq"][b, 0], batch["vel_seq"][b, 0],
                    batch["node_type"][b], batch["radius"][b],
                    n_steps=R, dt=dt,
                )
                pred_pos = torch.stack([s["pos"] for s in traj], dim=0)  # (R,N,3)
                gt_pos = batch["pos_seq"][b, 1:R + 1]
                total += F.mse_loss(pred_pos, gt_pos).item()
                n_samples += 1

        val = total / max(n_samples, 1)
        return {"total": val, "position": val, "rollout_pos_mse": val}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint (PhysicsNeMo-compatible format)."""
        if not self.is_master:
            return

        save_dir = Path(self.config.checkpoint["save_dir"])
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "metrics_history": self.metrics_history,
            "best_val_loss": self.best_val_loss,
        }

        # Regular checkpoint
        path = save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved: {path}")

        # Best checkpoint
        if is_best:
            best_path = save_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            logger.info(f"Best model saved: {best_path} (val_loss={self.best_val_loss:.6f})")

        # Latest for resume
        latest_path = save_dir / "latest.pt"
        torch.save(ckpt, latest_path)

    def load_checkpoint(self, path: Optional[str] = None):
        """Resume training from checkpoint."""
        if path is None:
            path = self.config.checkpoint.get("resume_dir")
        if path is None:
            return

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self._raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.metrics_history = ckpt.get("metrics_history", {})
        logger.info(
            f"Resumed from {ckpt_path} (epoch={ckpt['epoch']}, "
            f"step={self.global_step})"
        )

    def train(self):
        """
        Main training loop — PhysicsNeMo `launch` equivalent.

        Handles:
        1. Data loading
        2. Epoch loop
        3. Validation
        4. Checkpointing
        5. Logging
        """
        # ── Data ──
        cfg = self.config.data
        if cfg.get("use_synthetic", True):
            from .data import SyntheticDEMData
            train_set = SyntheticDEMData(
                n_samples=cfg.get("n_train", 200),
                n_particles=cfg.get("n_particles", 400),
                n_steps=cfg.get("n_steps", 120),
                dt=cfg.get("dt", 0.003),
                substeps=cfg.get("substeps", 15),
                cache_dir=cfg.get("cache_dir", None),
                seed=42,
            )
            val_set = SyntheticDEMData(
                n_samples=cfg.get("n_val", 20),
                n_particles=cfg.get("n_particles", 400),
                n_steps=cfg.get("n_steps", 120),
                dt=cfg.get("dt", 0.003),
                substeps=cfg.get("substeps", 15),
                cache_dir=cfg.get("cache_dir", None),
                seed=123,
            )
        else:
            from .data import DEMDataset
            train_set = DEMDataset(cfg["data_dir"], split="train")
            val_set = DEMDataset(cfg["data_dir"], split="val")

        # Shard the training set across ranks (genuine 4x data parallelism;
        # without this every rank would iterate the full dataset redundantly).
        self._train_sampler = (
            DistributedSampler(train_set, shuffle=True)
            if self.dist.distributed else None
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.training["batch_size"],
            shuffle=(self._train_sampler is None),
            sampler=self._train_sampler,
            collate_fn=collate_graphs,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.training["batch_size"],
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=2,
            pin_memory=True,
        )

        self.dist.print(
            f"\n{'='*60}\n"
            f"  Trace Training\n"
            f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params\n"
            f"  Data: {len(train_set)} train / {len(val_set)} val samples\n"
            f"  Device: {self.device}\n"
            f"  Distributed: {self.dist.distributed} "
            f"(world_size={self.dist.world_size})\n"
            f"{'='*60}\n"
        )

        # ── Fit & freeze the acceleration normalizer (CRITICAL) ──
        # Without this the loss is dominated by the constant gravity term and
        # plateaus immediately. Computed identically on every rank from the same
        # seeded dataset, so DDP needs no broadcast.
        if not bool(self._raw_model.stats_fitted.item()):
            dt = self.config.data["dt"]
            src = getattr(train_set, "data", None)
            if src is not None and len(src) > 0:
                accs = []
                for s in src:
                    v = s["vel_seq"]                     # (T, N, 3)
                    accs.append(((v[1:] - v[:-1]) / dt).reshape(-1, 3))
                self._raw_model.accel_norm.fit(torch.cat(accs, dim=0))
                self._raw_model.stats_fitted.fill_(1.0)
                self.dist.print(
                    f"  Accel norm fitted: mean="
                    f"{[round(x, 3) for x in self._raw_model.accel_norm.mean.tolist()]} "
                    f"std={[round(x, 3) for x in self._raw_model.accel_norm.std.tolist()]}"
                )

        # ── Resume if specified ──
        self.load_checkpoint()

        # ── Training loop ──
        n_epochs = self.config.training["epochs"]
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, n_epochs):
            epoch_start = time.time()
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)  # reshuffle shards each epoch

            # Train
            train_losses = self.train_epoch(train_loader, epoch)

            # Validate
            val_losses = self.validate(val_loader)

            # (LR scheduler is stepped per optimizer step inside train_epoch.)

            # Track
            self.metrics_history["train_loss"].append(train_losses["total"])
            self.metrics_history["val_loss"].append(val_losses["total"])
            self.metrics_history["lr"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Check for best
            is_best = val_losses["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total"]

            # Log epoch summary
            epoch_time = time.time() - epoch_start
            self.dist.print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_losses['total']:.6f} | "
                f"Val Loss: {val_losses['total']:.6f} | "
                f"Best: {self.best_val_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch/train_loss": train_losses["total"],
                    "epoch/val_loss": val_losses["total"],
                    "epoch/lr": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                })

            # Save checkpoint
            if self.is_master and (
                epoch % self.config.checkpoint.get("save_interval", 10) == 0
                or epoch == n_epochs - 1
                or is_best
            ):
                self.save_checkpoint(epoch, is_best=is_best)

        # ── Final save ──
        self.dist.print(f"\nTraining complete! Best val loss: {self.best_val_loss:.6f}")
        if self.is_master:
            self.save_checkpoint(n_epochs - 1)

        # ── Save metrics ──
        if self.is_master:
            metrics_path = Path(self.config.logging["log_dir"]) / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)

        # ── Cleanup ──
        if self.use_wandb:
            import wandb
            wandb.finish()


# ═══════════════════════════════════════════════════════════════════
# Convenience: import collate_graphs here for cleaner user code
# ═══════════════════════════════════════════════════════════════════

from .data import collate_graphs  # noqa: E402, F811
