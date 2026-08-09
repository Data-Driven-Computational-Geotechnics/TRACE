#!/usr/bin/env python3
"""Trainer for the TRACE method (this package) on a Sand dataset.

Trace carries a learnable memory on each CONTACT EDGE across steps. Training uses
multi-step truncated BPTT over a window K (carrying the edge memory), a
normalized-acceleration Huber loss and GNS input noise; the best checkpoint is
chosen by short-rollout position MSE on the validation set.

Dataset-agnostic: the spatial dimension comes from the dataset metadata, so this
serves both 2D (008-Sand) and 3D (009-Sand-3D). Usually invoked from an
experiment directory via its run.sh / config, e.g.:
    python model/01-trace/train.py --config trace/config/trace.yaml
Outputs land in <results_path>/trace/results/<exp_name>/ (see common/trainer_common.py).
"""
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 抗碎片化, 须在 torch 导入前
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint as _grad_ckpt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # this package (tracegnn)
sys.path.insert(0, str(HERE.parent / "common"))     # trainer_common, sand_data
import trainer_common as tc


def pick_model(dim):
    if dim == 2:
        from tracegnn.model2d import Trace2D, Trace2D_NoMemory
        return Trace2D, Trace2D_NoMemory
    from tracegnn.model3d import Trace, Trace_NoMemory
    return Trace, Trace_NoMemory


def main():
    cfg = tc.load_config("trace")
    seed = int(cfg["seed"]); torch.manual_seed(seed)

    # ★多卡数据并行(torchrun 启动时自动启用)★: 样本按卡分片, 梯度手动 all-reduce 平均。
    # 不用 DDP 包装器: 与逐步梯度检查点/K步多次前向的组合更简单可靠(模型仅~2M参数, 同步开销可忽略)。
    ddp = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    if ddp:
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    else:
        dist = None; local_rank = 0

    if rank == 0:
        exp_name, ckpt_dir, log = tc.setup_run(cfg)
    else:
        import logging
        exp_name, ckpt_dir = cfg.get("exp_name", "run"), None
        log = logging.getLogger(f"trace.rank{rank}"); log.addHandler(logging.NullHandler()); log.propagate = False
    dev = torch.device("cuda", local_rank) if ddp else tc.get_device(cfg, log)

    train, val, info = tc.load_dataset(cfg)
    dim, radius, skin, box_size = info["dim"], info["radius"], info["skin"], info["box_size"]
    dt = 1.0  # GNS displacement convention

    memory = cfg["memory"]
    K, epochs = int(cfg["rollout_steps"]), int(cfg["epochs"])
    val_rollout, noise, lr = int(cfg["val_rollout"]), float(cfg["noise"]), float(cfg["lr"])
    normalize_inputs = bool(cfg.get("normalize_inputs", False))
    boundary_features = bool(cfg.get("boundary_features", False))
    sample_weighting = cfg.get("sample_weighting", "uniform")
    loss_type = cfg.get("loss", "huber")          # "huber"(default) | "mse" | "huberK"(delta=K)
    huber_delta = float(cfg.get("huber_delta", 1.0))

    def accel_loss(pred, tgt):
        if loss_type == "mse":
            return torch.nn.functional.mse_loss(pred, tgt)
        return torch.nn.functional.huber_loss(pred, tgt, delta=huber_delta)

    # ★长时漂移修复★: rollout 训练模式——训练时喂模型自己的预测(而非真值)滚 K 步, 对长程
    # 位置算 loss, 把"自我纠偏"直接训进去。teacher(默认)=每步喂真值(现状)。
    train_mode = cfg.get("train_mode", "teacher")            # "teacher" | "rollout"
    # ★P1 噪声校正目标(GNS 式)★: teacher 模式下目标从加噪态反推——从 (p̃,ṽ) 半隐式积分
    # 一步应精确落回干净轨迹 pos[t+1]，即 a* = ((pos[t+1]-p̃)/dt - ṽ)/dt。旧写法目标用干净
    # vel 算，模型只学"抗噪"不学"纠偏"，噪声/rollout 造成的位置偏移永不回收 → 长时漂移。
    target_correction = bool(cfg.get("target_correction", False))
    # ★P2 训推一致★: rollout 训练的漂移段/grad 段施加与推理 rollout 相同的硬约束
    # (穿透投影+边界 clamp)，消除"训练见裸漂移态、推理见投影态"的分布错配。
    constrained_rollout = bool(cfg.get("constrained_rollout", False))
    # ★P3 位移速度★: 约束后用位移重算速度 v=(pos-prev)/dt，恢复数据约定 vel[t]=Δpos。
    vel_from_displacement = bool(cfg.get("vel_from_displacement", False))
    pos_loss_w = float(cfg.get("pos_loss_weight", 300.0))    # 位置loss权重(loss很小,需放大)
    rollout_warmup = int(cfg.get("rollout_warmup_epochs", 0))  # curriculum: 漂移上限从0涨到max
    rollout_drift_max = int(cfg.get("rollout_drift_max", 40))  # pushforward: no_grad 漂移上限步数
    density_loss_w = float(cfg.get("density_loss_weight", 0.0))  # ★凝聚性★: 匹配局部密度(抑过散/结块)
    R_conn = skin * 2.0 * radius                              # 连接半径, 密度核尺度

    def density_loss(pred_pos, true_pos, rr):
        # 软局部密度 = Σ_j exp(-(d_ij/rr)²); 逐粒子匹配 pred vs true 的密度(相对MSE)。
        # 过散→预测密度偏低; 结块→局部密度偏高; 两者都被惩罚 → 逼出正确凝聚形态。
        dp = torch.exp(-(torch.cdist(pred_pos, pred_pos) / rr) ** 2).sum(1)
        dt = torch.exp(-(torch.cdist(true_pos, true_pos) / rr) ** 2).sum(1)
        return torch.nn.functional.mse_loss(dp, dt) / (dt.mean() ** 2 + 1e-6)

    Trace, Trace_NoMemory = pick_model(dim)
    Cls = Trace_NoMemory if memory == "none" else Trace
    model = Cls(hidden_dim=int(cfg["hidden_dim"]), memory_dim=int(cfg["memory_dim"]),
                num_layers=int(cfg["num_layers"]), skin_factor=skin, memory_type=memory,
                normalize_inputs=normalize_inputs, boundary_features=boundary_features).to(dev)
    model.noise_std = noise
    model.vel_from_displacement = vel_from_displacement  # P3(影响 rollout/验证/推理)
    R_clip = skin * 2.0 * radius                         # 到墙距离裁剪半径 = 连接半径
    if boundary_features:                                # 钉进 buffer(rollout 入口会再校准)
        model.box_size.fill_(float(box_size))
        model.feat_clip.fill_(float(R_clip))
    model.set_accel_stats(*tc.fit_accel_stats(train, dt, dim))
    if normalize_inputs:                                 # 在【加噪后】分布上拟合输入统计量
        vmean, vstd = tc.fit_vel_stats(train, dim)
        estd = tc.fit_edge_stats(model, train, dim, noise, device=dev)
        model.set_input_stats(vmean, vstd, estd)

    # ★从已训好的 checkpoint 微调★(rollout 训练标准 recipe: teacher 预训练→rollout 微调)。
    # 架构必须与 checkpoint 一致(layers/normalize_inputs/boundary_features)。
    init_from = cfg.get("init_from")
    if init_from:
        _ck = torch.load(init_from, map_location=dev, weights_only=False)
        model.load_state_dict(_ck["model_state_dict"])
        log.info(f"init    | 从 {init_from} 加载预训练权重微调")

    if ddp:                                             # 广播 rank0 权重, 各卡起点严格一致
        for _prm in model.state_dict().values():
            if torch.is_tensor(_prm) and _prm.is_floating_point():
                dist.broadcast(_prm, src=0)

    # ★梯度检查点★: 逐 BPTT 步重算激活, 峰值显存从 O(K) 降到 O(1) 步(3D 必需, 代价~1.5x前向)
    use_ckpt = bool(cfg.get("grad_checkpoint", False))

    log.info("═" * 64)
    log.info(f"TRACE (edge memory) · {dim}D sand · exp={exp_name}")
    log.info(f"config  | memory={memory}  hidden={cfg['hidden_dim']}  mem_dim={cfg['memory_dim']}  "
             f"layers={cfg['num_layers']}  K={K}  noise={noise:.1e}  lr={lr:.1e}  seed={seed}")
    log.info(f"fixes   | target_correction={target_correction}(P1)  "
             f"constrained_rollout={constrained_rollout}(P2)  "
             f"vel_from_displacement={vel_from_displacement}(P3)")
    log.info(f"inputs  | normalize_inputs={normalize_inputs}  boundary_features={boundary_features}  "
             f"sample_weighting={sample_weighting}"
             + (f"  | vel_std={[round(x,5) for x in model.vel_norm.std.tolist()]}  "
                f"edge_std={[round(x,5) for x in model.edge_norm.std.tolist()]}" if normalize_inputs else ""))
    log.info(f"data    | dim={dim}  N_train={len(train)}  N_val={len(val)}  radius={radius}  "
             f"skin={skin:.3f}  conn={2*radius*skin:.4f}  box={box_size:.3f}")
    log.info(f"model   | params={sum(p.numel() for p in model.parameters()):,}  "
             f"accel_std={[round(x,5) for x in model.accel_norm.std.tolist()]}")
    decay_epochs = int(cfg.get("decay_epochs") or epochs)
    lr_min = float(cfg.get("lr_min", lr * 0.01)); patience = int(cfg.get("early_stop_patience", 0))
    log.info(f"run     | epochs={epochs}  decay_epochs={decay_epochs}  lr_min={lr_min:.1e}  "
             f"early_stop={patience or 'off'}  val_rollout={val_rollout}  ->  {ckpt_dir}")
    log.info(f"metrics | loss(accelHuber)=优化目标:归一化加速度 Huber(无单位)  |  "
             f"RMSE@{val_rollout}=位置误差 over {val_rollout}步 rollout(域单位,↓越好)  |  "
             f"选 checkpoint 依据 valRMSE")
    log.info("─" * 64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    _wpe = int(cfg.get("windows_per_epoch", 1))
    steps_per_ep = (len(train) * _wpe // world) if ddp else len(train) * _wpe
    sched = tc.make_scheduler(opt, cfg, max(1, steps_per_ep))

    # 昂贵指标：位置 RMSE over rollout —— 选 checkpoint 的依据(域单位,越小越好)。
    # 每条验证轨迹要跑 val_rollout 步自回归,很贵 → 只在 eval_interval 的 epoch 才算。
    def rollout_rmse(samples, horizon):
        tot_mse = 0.0; n = 0
        my = samples[rank::world] if ddp else samples          # 验证轨迹按卡分片
        for s in my:
            pos = torch.as_tensor(s["pos"]).to(dev); vel = torch.as_tensor(s["vel"]).to(dev)
            nt = torch.as_tensor(s["node_type"]).to(dev); rad = torch.as_tensor(s["radius"]).to(dev)
            # ★从 frame1 用真实初速度 vel[1] 播种★：vel[0] 被 loader 置 0，从它起步会丢掉
            # 各土堆的初动量(方向)，rollout 只剩重力→退化成自由落体。选模型必须用对的协议。
            R = min(horizon, pos.shape[0] - 2)
            traj = model.rollout(pos[1], vel[1], nt, rad, n_steps=R, dt=dt, box_size=box_size)
            pred = torch.stack([t["pos"] for t in traj], 0)        # 预测 frames 2..R+1
            tot_mse += torch.nn.functional.mse_loss(pred, pos[2:R + 2]).item(); n += 1
        if ddp:                                                # 汇总各卡分片
            agg = torch.tensor([tot_mse, float(n)], device=dev)
            dist.all_reduce(agg)
            tot_mse, n = float(agg[0]), int(agg[1])
        return (tot_mse / max(n, 1)) ** 0.5

    # 便宜指标：和训练同款的"归一化加速度 Huber"(K步窗口,无噪声,无 rollout)。每个 epoch
    # 都算,用来看收敛/过拟合(train vs val 同尺),不用于选 checkpoint。比 rollout 便宜~20×。
    def val_onestep_loss(samples):
        tot = 0.0; nb = 0
        for s in samples:
            pos = torch.as_tensor(s["pos"]).to(dev); vel = torch.as_tensor(s["vel"]).to(dev)
            nt = torch.as_tensor(s["node_type"]).to(dev); rad = torch.as_tensor(s["radius"]).to(dev)
            T = pos.shape[0]; hi = T - 1 - K
            if hi < 2:
                continue
            ts = int(torch.randint(1, hi, (1,)).item())
            mem = None; idm = None; loss = 0.0
            for k in range(K):
                tgt = model.accel_norm.normalize((vel[ts + k + 1] - vel[ts + k]) / dt)
                o = model(pos=pos[ts + k], vel=vel[ts + k], node_type=nt, radius=rad,
                          edge_memory_state=mem, edge_id_map=idm)
                loss = loss + accel_loss(o["accel"], tgt)
                mem = o["edge_memory_state"]; idm = o["edge_id_map"]
            tot += (loss / K).item(); nb += 1
        return tot / max(nb, 1)

    # eval_interval: 每隔多少 epoch 才做一次昂贵的 rollout 验证(默认 1=每 epoch,向后兼容旧 config)
    eval_interval = max(1, int(cfg.get("eval_interval", 1)))
    log.info(f"eval    | eval_interval={eval_interval}(每{eval_interval}epoch做一次长时rollout验证选checkpoint; "
             f"每epoch算便宜的 valLoss(one-step) 看收敛)")

    history = []; best = float("inf"); best_ep = -1; t0 = time.time()
    last_val_rmse = float("nan"); last_train_rmse = float("nan")
    for ep in range(epochs):
        t_ep = time.time(); model.train(); tot = 0.0; nb = 0
        g = torch.Generator().manual_seed(seed * 100003 + ep)   # 各卡同 perm(确定性)
        wpe = int(cfg.get("windows_per_epoch", 1))               # 每条轨迹每 epoch 采样的窗口数
        pool = list(range(len(train))) * wpe
        perm = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
        if ddp:
            perm = perm[: (len(perm) // world) * world]         # 保证各卡步数一致
            perm = perm[rank::world]
        for si in perm:
            s = train[si]
            pos = torch.as_tensor(s["pos"]).to(dev); vel = torch.as_tensor(s["vel"]).to(dev)
            nt = torch.as_tensor(s["node_type"]).to(dev); rad = torch.as_tensor(s["radius"]).to(dev)
            T = pos.shape[0]
            margin = (K + rollout_drift_max) if train_mode == "rollout" else K  # rollout 要留 drift 余量
            hi = T - 1 - margin                        # ts ∈ [1, hi-1] → 最大访问帧 ≤ T-1 不越界
            if hi < 2:                                 # 轨迹太短无合法窗口 → 跳过,防 IndexError
                continue
            if sample_weighting == "motion":
                # 按每帧平均速度加权采样窗口起点(只在合法起点范围内)，避免大量窗口落在
                # 静止沙堆(目标加速度≈0)而稀释信号；权重平移到非负。
                speeds = vel[1:hi].reshape(hi - 1, -1, dim).norm(dim=-1).mean(dim=1)  # (hi-1,)
                w = speeds - speeds.min() + 1e-8
                ts = 1 + int(torch.multinomial(w, 1).item())
            else:
                ts = int(torch.randint(1, hi, (1,)).item())
            mem = None; idm = None; opt.zero_grad()
            if train_mode == "rollout":
                # ★pushforward 训练(内存高效)★: 先 no_grad 漂移 d 步(便宜)到长程漂移态, 再
                # grad-rollout K 步算位置 loss → 让模型学会从"自己漂移的状态"纠偏回真值。
                # 只用位置 loss(漂移态对不上真值加速度, accel loss 会爆, 见 smoke)。
                # curriculum: 漂移上限 d_max 从 0 逐步涨到 rollout_drift_max。
                d_max = rollout_drift_max if rollout_warmup <= 0 else \
                        int(round(rollout_drift_max * min(1.0, ep / rollout_warmup)))
                d = int(torch.randint(0, d_max + 1, (1,)).item())   # 本次随机漂移长度
                cur_pos = pos[ts] + torch.randn_like(pos[ts]) * noise
                cur_vel = vel[ts] + torch.randn_like(vel[ts]) * noise
                edge_cap = 20 * pos.shape[1]                         # 边数护栏(正常~7N, 3倍余量)
                with torch.no_grad():                                # 漂移段不回传梯度(省内存)
                    for k in range(d):
                        o = model(pos=cur_pos, vel=cur_vel, node_type=nt, radius=rad,
                                  edge_memory_state=mem, edge_id_map=idm)
                        prev_pos = cur_pos
                        cur_vel = cur_vel + o["accel_phys"] * dt; cur_pos = cur_pos + cur_vel * dt
                        if constrained_rollout:                      # ★P2: 与推理同款硬约束★
                            cur_pos, cur_vel = model._constrain_state(
                                cur_pos, cur_vel, prev_pos, rad, box_size, dt)
                        mem = o["edge_memory_state"]; idm = o["edge_id_map"]
                        if mem is not None and torch.is_tensor(mem) and mem.shape[0] > edge_cap:
                            break                                    # 状态病态(边数爆炸)→截断漂移, 样本仍有效
                cur_pos = cur_pos.detach(); cur_vel = cur_vel.detach()
                if torch.is_tensor(mem): mem = mem.detach()
                ploss = 0.0; dloss = 0.0                             # grad-rollout K 步
                side = {}
                k_done = 0
                for k in range(K):
                    def _step(a, b, c, _idm=idm):
                        o = model(pos=a, vel=b, node_type=nt, radius=rad,
                                  edge_memory_state=c, edge_id_map=_idm)
                        side["idm"] = o["edge_id_map"]
                        return o["accel_phys"], o["edge_memory_state"]
                    if use_ckpt:
                        accel_phys, new_mem = _grad_ckpt(_step, cur_pos, cur_vel, mem, use_reentrant=False)
                    else:
                        accel_phys, new_mem = _step(cur_pos, cur_vel, mem)
                    prev_pos = cur_pos
                    cur_vel = cur_vel + accel_phys * dt               # ★用自己的预测推进★
                    cur_pos = cur_pos + cur_vel * dt
                    if constrained_rollout:
                        # ★P2 直通估计(STE)★: 前向值与推理一致(投影+clamp+P3 位移速度)，
                        # 梯度按恒等映射穿过约束(约束本身不可微)。传 detach 的副本，
                        # 避免 _constrain_state 的原地写破坏 autograd 图。
                        cpos, cvel = model._constrain_state(
                            cur_pos.detach().clone(), cur_vel.detach().clone(),
                            prev_pos.detach(), rad, box_size, dt)
                        cur_pos = cur_pos + (cpos - cur_pos).detach()
                        cur_vel = cur_vel + (cvel - cur_vel).detach()
                    ploss = ploss + torch.nn.functional.mse_loss(cur_pos, pos[ts + d + k + 1]) / (box_size ** 2)
                    if density_loss_w > 0:                            # ★凝聚性 loss(抑过散/结块)★
                        dloss = dloss + density_loss(cur_pos, pos[ts + d + k + 1], R_conn)
                    mem = new_mem; idm = side["idm"]
                    k_done += 1
                    if torch.is_tensor(new_mem) and new_mem.shape[0] > edge_cap:
                        break                                        # 边数爆炸→提前收束
                loss = (pos_loss_w * ploss + density_loss_w * dloss) / max(k_done, 1)
            else:
                loss = 0.0
                side = {}
                for k in range(K):
                    pk = pos[ts + k] + torch.randn_like(pos[ts + k]) * noise
                    vk = vel[ts + k] + torch.randn_like(vel[ts + k]) * noise
                    if target_correction:
                        # ★P1★ 目标从加噪态反推(noise=0 时精确退化为旧目标):
                        # 半隐式积分 v'=ṽ+a·dt, p'=p̃+v'·dt 要求 p'=pos[t+1]
                        #   → a* = ((pos[t+1] - p̃)/dt - ṽ)/dt
                        tgt_phys = ((pos[ts + k + 1] - pk) / dt - vk) / dt
                    else:
                        tgt_phys = (vel[ts + k + 1] - vel[ts + k]) / dt
                    tgt = model.accel_norm.normalize(tgt_phys)

                    def _step(a, b, c, _idm=idm):
                        o = model(pos=a, vel=b, node_type=nt, radius=rad,
                                  edge_memory_state=c, edge_id_map=_idm)
                        side["idm"] = o["edge_id_map"]
                        return o["accel"], o["edge_memory_state"]
                    if use_ckpt:
                        accel, mem = _grad_ckpt(_step, pk, vk, mem, use_reentrant=False)
                    else:
                        accel, mem = _step(pk, vk, mem)
                    idm = side["idm"]
                    loss = loss + accel_loss(accel, tgt)
                loss = loss / K
            loss.backward()
            if ddp:                                              # 手动梯度 all-reduce 平均
                for _prm in model.parameters():
                    if _prm.grad is not None:
                        dist.all_reduce(_prm.grad)
                        _prm.grad.div_(world)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
        trl = tot / max(nb, 1)

        model.eval()
        do_rollout = (ep % eval_interval == 0) or (ep == epochs - 1)   # 昂贵验证只在间隔点
        improved = False
        with torch.no_grad():
            val_loss = val_onestep_loss(val[:4])                  # 便宜: 每 epoch 都算
            if do_rollout:
                last_val_rmse = rollout_rmse(val, val_rollout)            # 验证集: 位置 RMSE @ val_rollout
                last_train_rmse = rollout_rmse(train[:4], val_rollout)    # 训练集(少量,同指标): 泛化差距
                improved = last_val_rmse < best
                if improved:
                    best, best_ep = last_val_rmse, ep
                    if rank != 0:
                        pass
                    else:
                        torch.save({"model_state_dict": model.state_dict(),
                                    "config": dict(model_name="trace", dim=dim, memory=memory, skin_factor=skin,
                                               radius=radius, box_size=box_size, dt=dt,
                                               hidden_dim=int(cfg["hidden_dim"]), memory_dim=int(cfg["memory_dim"]),
                                               num_layers=int(cfg["num_layers"]),
                                               normalize_inputs=normalize_inputs,
                                               boundary_features=boundary_features,
                                               vel_from_displacement=vel_from_displacement,
                                               target_correction=target_correction,
                                               constrained_rollout=constrained_rollout)},
                               ckpt_dir / "best_model.pt")
        cur_lr = sched.get_last_lr()[0]; sec = time.time() - t_ep
        # history 里 rollout 指标用最近一次的值(carry-forward),保证曲线连续
        history.append(dict(epoch=ep, loss=trl, val_loss=val_loss,
                            train_rmse=last_train_rmse, val_rmse=last_val_rmse,
                            train=last_train_rmse, val=last_val_rmse,
                            lr=cur_lr, sec=round(sec, 2), best=improved, rolled=do_rollout))
        if do_rollout:
            log.info(f"epoch {ep:4d}/{epochs} | loss {trl:7.4f} | valLoss {val_loss:7.4f} | "
                     f"trainRMSE@{val_rollout} {last_train_rmse:.3e} | valRMSE@{val_rollout} {last_val_rmse:.3e} | "
                     f"best {best:.3e} {'★' if improved else ' '} | lr {cur_lr:.2e} | {sec:5.1f}s")
        else:
            log.info(f"epoch {ep:4d}/{epochs} | loss {trl:7.4f} | valLoss {val_loss:7.4f} | "
                     f"(rollout skipped) | best {best:.3e} | lr {cur_lr:.2e} | {sec:5.1f}s")
        # 早停: 只在做了 rollout 的 epoch 判断(patience 单位仍是 epoch)
        if patience and do_rollout and best_ep >= 0 and (ep - best_ep) >= patience:
            log.info("─" * 64)
            log.info(f"EARLY STOP | valRMSE not improved for {patience} epochs "
                     f"(best valRMSE {best:.3e} @ epoch {best_ep})")
            break

    elapsed = (time.time() - t0) / 60.0
    if ddp:
        dist.barrier()                                   # 全员对齐后立即销毁通信组——
        dist.destroy_process_group()                     # rank0 的终评长达数十分钟, 不能留通信组挂着
    if rank != 0:                                        # 非主卡到此结束
        return
    tc.plot_curves(history, ckpt_dir, exp_name)

    # ── comprehensive evaluation ──
    import eval_common as ec
    log.info("─" * 64)
    log.info("EVAL   | computing rollout metrics (RMSE, KE, overlap, deposit, momentum)...")
    # n_rollout=320: 覆盖整条轨迹(320帧)——旧值120只量了前1/3, 长时质量根本没被测到
    metrics = ec.run_eval(model, val, dev, dim, dt, box_size, radius,
                          n_rollout=320, n_long=320, save_dir=ckpt_dir / "eval")
    s = metrics["summary"]
    log.info(f"eval   | rmse_end={s['rmse_end']:.3e}  ke_end={s['ke_pred_end']:.3e}(vs gt {s['ke_gt_end']:.3e})  "
             f"overlap_max={s['overlap_max']*100:.1f}%  mom_resid={s['mom_resid']:.3e}  long_ke={s['long_ke_final']:.3e}")
    log.info(f"deposit| runout pred={s['deposit']['runout_pred_mean']:.3f} vs GT={s['deposit']['runout_gt_mean']:.3f}  "
             f"height pred={s['deposit']['height_pred_mean']:.3f} vs GT={s['deposit']['height_gt_mean']:.3f}")
    log.info(f"figures| {ckpt_dir / 'eval'}/")

    log.info("─" * 64)
    log.info(f"DONE | best_valRMSE={best:.3e} @ epoch {best_ep} | elapsed {elapsed:.1f} min | "
             f"ckpt={ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
