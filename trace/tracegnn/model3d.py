"""
================================================================================
Trace — Contact-Memory Graph Networks（接触记忆图网络）主模型
================================================================================

一句话概括：
  一个用 GNN 学习颗粒离散元（DEM）物理的模拟器。与传统 GNS 的唯一区别是：
  每条接触"边"上携带一个可学习的 GRU 记忆向量，在时间步之间跨步传递。

三层架构：
  1. Encoder（编码器）：粒子状态/边特征 → 128 维隐含空间
  2. Processor（处理器）：K 层消息传递，每条边的消息拼接了边记忆
  3. Decoder（解码器）：隐含状态 → 加速度（外部 + 接触力）

加速度分解（核心公式）：
  a_i = external_i                      ← 重力、边界力（每个粒子独立学习，线性输出头）
      + (1/m_i) * Σ_j f_ij             ← 内部接触力（反对称：+f 给 i，-f 给 j）
                                          → 自动保证线性动量守恒（牛顿第三定律）

与初版代码的关键修复：
  - EdgeMemoryState 现在是 nn.Module → 其 GRU/birth_init 参数会被 optimizer 训练、能被 .to(device) 移动
  - node_decoder 是 LINEAR 输出头（final_act=False）→ 可以输出负值（向下加速度，如重力）
    旧代码用 SiLU 结尾 → 输出 ≥ -0.278 → 永远学不会向下的重力
  - 边记忆在每一步只更新一次（不是每层消息传递都更新）→ 物理上合理
  - 加速度目标/输出通过 Normalizer 做 z-score 归一化 → loss 量级稳定，不受数据尺度影响
================================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 导入依赖
# ═══════════════════════════════════════════════════════════════════════════════

import torch                                # PyTorch 核心库：张量运算、自动求导
import torch.nn as nn                       # PyTorch 神经网络模块：nn.Module、各类层
from typing import Tuple, Optional, Dict, List  # 类型注解：提升代码可读性

# 从同目录的 layers.py 导入自定义子模块
from .layers import (
    MLP,                        # 通用两层 MLP（SiLU + LayerNorm）
    RMSNorm,                    # Root Mean Square 归一化（比 LayerNorm 更轻量）
    Normalizer,                 # 加速度 z-score 归一化器（训练前 fit，训练中冻结）
    EdgeMemoryState,            # 边记忆的 GRU 版本（时间维度门控更新）
    EdgeSpatioTemporalMemory,   # 边记忆的"空间注意力 + GRU"版本
    EdgeSpatialOnly,            # 纯空间(消融): 每步线图注意力, 无时间携带
    EdgeSpatioTemporalAttn,     # 边记忆的"时空联合注意力"版本
    MemoryEdgeBlock,            # 感知边记忆的消息传递块
    ContactForceDecoder,        # 从边记忆解码接触力（法向 + Coulomb 夹紧切向）
)


# ═══════════════════════════════════════════════════════════════════════════════
# Trace 主模型类
# ═══════════════════════════════════════════════════════════════════════════════

class Trace(nn.Module):                              # 继承 nn.Module：注册参数、支持 .to(device)、.train()/.eval()
    """Contact-Memory Graph Networks for granular DEM simulation."""

    # ------------------------------------------------------------------
    # 构造函数：定义模型的所有组件
    # ------------------------------------------------------------------
    def __init__(
        self,
        dim: int = 3,                   # 空间维度 d∈{2,3}；决定输入/输出/边界各维度
        node_in_dim: int = None,        # 节点原始输入维度（默认 2d+2）
                                        # = pos(d) + vel(d) + radius(1) + node_type(1)
        edge_in_dim: int = None,        # 边原始输入维度（默认 2d+1）
                                        # = rel_pos(d) + rel_vel(d) + dist(1)
        hidden_dim: int = 128,          # 全网络的隐含层维度（编码器/处理器/解码器统一）
        memory_dim: int = 16,           # 边记忆向量的维度（m_ij ∈ R^16）
        num_layers: int = 10,           # Processor 中消息传递的层数 K
        num_history: int = 4,           # 历史速度帧数（GNS 的训练技巧，但当前 demo 不依赖）
        output_accel: bool = True,      # True=输出加速度，False=输出下一时刻速度
        skin_factor: float = 1.25,      # 建图的预接触因子（DEM skin factor），无物理量纲
                                        # 边(i,j)存在当：dist < skin_factor × (r_i + r_j)
                                        # 1.0=仅物理接触时建边，1.25=允许25%预接触缓冲
                                        # 这是粒子半径的确定性函数，非自由超参数，所有实验固定
        memory_type: str = "gru",       # 边记忆的类型：
                                        #   "gru"  = 纯时间 GRU（每个边独立，O(E)）
                                        #   "st"   = 空间线图注意力(1头) + 时间 GRU
                                        #   "stmh" = 空间线图注意力(4头) + 时间 GRU
                                        #   "st2"  = 时空联合注意力（窗口化时间注意力 + 空间注意力）
        normalize_inputs: bool = False,  # NEW: z-score 节点速度 + scale-only 归一化边特征
        boundary_features: bool = False, # NEW: 用"到墙裁剪距离"替换绝对位置作为节点特征
    ):
        super().__init__()                          # 必须先调用父类 nn.Module 的构造函数

        # ── 保存超参数为实例属性，供 forward/_build_graph 等函数使用 ──
        self.hidden_dim = hidden_dim                # 128：隐含特征维度
        self.memory_dim = memory_dim                # 16：每条边的记忆维度
        self.num_layers = num_layers                # K=10：消息传递层数
        self.num_history = num_history              # 4：历史帧数（GNS 风格）
        self.output_accel = output_accel            # True：输出加速度而非速度
        self.dim = dim                              # d∈{2,3}：空间维度
        # 未显式指定时按 d 推导：
        #   boundary_features=False → node=2d+2 (pos + vel + r + type)
        #   boundary_features=True  → node=3d+2 (用 2d 维"到墙距离" 替换 pos(d))
        #   edge=2d+1 (rel_pos+rel_vel+dist)，两个开关都不改变它
        self.normalize_inputs = normalize_inputs
        self.boundary_features = boundary_features
        if node_in_dim is None:
            node_in_dim = (3 * dim + 2) if boundary_features else (2 * dim + 2)
        if edge_in_dim is None:
            edge_in_dim = 2 * dim + 1
        self.node_in_dim = node_in_dim              # 节点输入特征数（d=3→8）
        self.edge_in_dim = edge_in_dim              # 边输入特征数（d=3→7）
        self.skin_factor = skin_factor              # 1.25：预接触因子（粒子半径的确定性函数）
        self.memory_type = memory_type              # "gru"：选择记忆方式
        # P3 开关：rollout 每步约束(投影/clamp)后用位移重算速度 v=(pos-prev_pos)/dt，
        # 保证模型输入始终满足数据约定 vel[t]=pos[t]-pos[t-1]（普通属性，不进 state_dict；
        # trainer 从 config 设置并存入 checkpoint config，工具脚本恢复）。
        self.vel_from_displacement = False

        # ═══════════════════════════ 编码器（Encoder）═══════════════════════════
        # 节点编码器：每个粒子的原始特征 → 128 维隐含向量
        # 输入维度 = node_in_dim(8)：pos(3) + vel(3) + radius(1) + type(1)
        # Trace 将时序记忆完全置于边 GRU 中，不使用节点历史速度缓冲
        self.node_encoder = MLP(node_in_dim, hidden_dim)
        # MLP(8, 128) 内部：Linear(8→128)→SiLU→LayerNorm→Linear(128→128)→SiLU

        # 边编码器：每条边的原始特征 → 128 维隐含向量
        self.edge_encoder = MLP(edge_in_dim, hidden_dim)
        # MLP(7, 128) 内部：Linear(7→128)→SiLU→LayerNorm→Linear(128→128)→SiLU

        # ═══════════════════════════ 边记忆模块 ═══════════════════════════
        # Trace 唯一的结构创新。根据 memory_type 选择不同的边记忆实现：
        #   "gru"  → EdgeMemoryState：纯 GRU 门控，O(E)，默认推荐
        #   "st"   → EdgeSpatioTemporalMemory：空间线图注意力(1头) + GRU
        #   "stmh" → EdgeSpatioTemporalMemory：空间线图注意力(4头) + GRU
        #   "st2"  → EdgeSpatioTemporalAttn：纯注意力（时间+空间），无 GRU
        if memory_type in ("st", "stmh"):
            # 空间注意力版本：每条边"看到"共享节点的邻居边，再通过 GRU 更新
            self.edge_memory = EdgeSpatioTemporalMemory(
                memory_dim=memory_dim,       # 16：GRU 的 hidden_size
                hidden_dim=hidden_dim,       # 128：GRU 的 input_size
                edge_in_dim=edge_in_dim,     # 7：birth_init MLP 的输入维度
                heads=(4 if memory_type == "stmh" else 1)  # 注意力头数
            )
        elif memory_type == "spatial":
            # 纯空间(消融)：每步线图注意力算 s_ij^t 喂解码器, 不跨步携带时间状态
            self.edge_memory = EdgeSpatialOnly(
                memory_dim=memory_dim, hidden_dim=hidden_dim, edge_in_dim=edge_in_dim
            )
        elif memory_type == "st2":
            # 纯注意力版本：窗口化时间自注意力 + 空间线图注意力
            self.edge_memory = EdgeSpatioTemporalAttn(
                memory_dim=memory_dim, hidden_dim=hidden_dim, edge_in_dim=edge_in_dim
            )
        else:
            # 纯 GRU 版本（默认，推荐）：每条边独立更新
            # 这是论文的核心 baseline——最简单的边记忆实现
            self.edge_memory = EdgeMemoryState(
                memory_dim=memory_dim,       # 16：GRU 的 hidden_size
                hidden_dim=hidden_dim,       # 128：GRU 的 input_size
                edge_in_dim=edge_in_dim      # 7：birth_init MLP 的输入维度
            )

        # ═══════════════════════════ 处理器（Processor）═══════════════════════════
        # nn.ModuleList 保证子模块的参数被 optimizer 跟踪、被 .to(device) 移动
        # K 个 MemoryEdgeBlock：每个是一个消息传递层
        # 边消息中拼接了 edge_mem → MLP → 聚合到节点 → 残差更新
        self.processor = nn.ModuleList([
            MemoryEdgeBlock(hidden_dim, hidden_dim, memory_dim)
            # 参数：(node_feat_dim=128, edge_feat_dim=128, memory_dim=16)
            for _ in range(num_layers)      # 重复 K=10 次
        ])

        # K 个 RMSNorm：每层消息传递后的节点层归一化（稳定训练）
        self.processor_norms = nn.ModuleList([
            RMSNorm(hidden_dim)             # 对 128 维向量做 RMS 归一化
            for _ in range(num_layers)
        ])

        # ═══════════════════════════ 解码器（Decoder）═══════════════════════════
        # 节点解码器：每个粒子的 128 维隐含状态 → 3 维外部加速度（如重力+边界力）
        # final_act=False：末尾没有 SiLU 激活函数 → 输出可以是任意实数
        # 这是关键修复！旧代码 final_act=True(SiLU) → 输出 ≥ -0.278 → 学不会向下的重力
        self.node_decoder = MLP(hidden_dim, dim, final_act=False)
        # MLP(128, 3, final_act=False) 内部：Linear(128→128)→SiLU→LayerNorm→Linear(128→3)
        # 输出就是 3 维加速度向量，无激活函数限制

        # 接触力解码器：每条边的记忆 + 边特征 → 成对接触力
        # 输入：edge_mem(E,16) + edge_h(E,128)
        # 输出：
        #   F_n(E,1) — 法向力（≥0，softplus 保证）
        #   F_t(E,3) — 切向力（Coulomb 锥夹紧：|F_t| ≤ μ·F_n）
        #   μ(E,1)   — 学习到的摩擦系数 ∈ [0.1, 1.0]
        self.force_decoder = ContactForceDecoder(
            memory_dim=memory_dim,          # 16：边记忆维度
            edge_dim=hidden_dim,            # 128：边隐含特征维度
            hidden_dim=hidden_dim,          # 128：内部 MLP 的隐含维度
            dim=dim,                        # d：切向力向量维度
        )

        # ═══════════════════════════ 加速度归一化器 ═══════════════════════════
        # 训练前在训练集上 fit 一次（计算加速度的全局均值和标准差），训练中冻结
        # 作用：将加速度目标标准化到 ~N(0,1) → loss 不受数据尺度影响 → 稳定训练
        self.accel_norm = Normalizer(dim)                   # d 维归一化器（每个空间通道一个）

        # register_buffer：不是可训练参数，但会被 state_dict() 保存、被 .to(device) 移动
        # stats_fitted=0 表示归一化器尚未 fit（训练开始前需要调用 set_accel_stats）
        self.register_buffer("stats_fitted", torch.zeros(1))

        # ═══════════════════════════ 输入归一化器（NEW）═══════════════════════════
        # 只在 normalize_inputs=True 时注册——这样旧 checkpoint(无这些 buffer)仍能
        # 以 strict=True 加载。vel 用 z-score；边特征 scale-only(mean=0)以保住双向负化。
        if normalize_inputs:
            self.vel_norm = Normalizer(dim, center=True)            # 节点速度 z-score
            self.edge_norm = Normalizer(2 * dim + 1, center=False)  # 边特征 scale-only
            self.register_buffer("input_stats_fitted", torch.zeros(1))

        # 边界特征所需的标量(盒边长、裁剪距离)——同样条件注册成 buffer，随 state_dict 保存
        if boundary_features:
            self.register_buffer("box_size", torch.zeros(1))   # 盒边长 L(rollout 入口会校准)
            self.register_buffer("feat_clip", torch.ones(1))   # 到墙距离裁剪半径 R_clip

        # ═══════════════════════ 输出头小初始化（关键）═══════════════════════
        # 让初始总加速度≈0。否则(尤其归一化输入后,内部特征是 O(1))未训练的解码器会
        # 输出 O(1) 物理力 → 初始 accel_phys≈O(1)，是 accel_std(~5e-4)的上千倍 → rollout
        # 爆炸、loss 从~1300 起且卡在~40(softplus 力难压小)。改为从~0 往上学(易)。
        self._init_output_heads()

    def _init_output_heads(self):
        """node_decoder 末层置零(ext_accel=0)；normal_head 置零+负 bias(F_n≈3e-4/边,
        接近物理力尺度而非 O(1))。F_t 受 mu·F_n 钳制,故 F_n 小则整条接触力小。"""
        with torch.no_grad():
            last = self.node_decoder.net[-1]              # MLP 末层 Linear(hidden→dim)
            last.weight.zero_(); last.bias.zero_()
            self.force_decoder.normal_head.weight.zero_()
            self.force_decoder.normal_head.bias.fill_(-8.0)   # softplus(-8)≈3.4e-4

    # ------------------------------------------------------------------
    # 设置输入归一化统计量（训练前调用一次；normalize_inputs=False 时为空操作）
    # ------------------------------------------------------------------
    def set_input_stats(self, vel_mean, vel_std, edge_std):
        """vel_norm 用 z-score；edge_norm 为 scale-only(mean 强制为 0)。

        edge_std 应是 trainer 在【加噪后】的边特征分布上拟合的 zero-centered std
        (sqrt(E[x²]))，宽度 2*dim+1。若在干净数据上拟合，rel_vel 通道 std~6e-8 会
        让归一化爆炸数千倍(见设计 spec D4)。"""
        if not self.normalize_inputs:
            return
        # 退化(近零方差)通道 → std=1.0(保持其自然~0尺度)，而非 clamp 到 eps=1e-6
        # (后者会把该通道的输入噪声放大上千倍，正是本次要消除的尺度病)。与
        # Normalizer.fit / set_accel_stats 的策略一致。
        vel_std = torch.where(vel_std < self.vel_norm.eps, torch.ones_like(vel_std), vel_std)
        edge_std = torch.where(edge_std < self.edge_norm.eps, torch.ones_like(edge_std), edge_std)
        self.vel_norm.mean.copy_(vel_mean.to(self.vel_norm.mean))
        self.vel_norm.std.copy_(vel_std.to(self.vel_norm.std))
        self.edge_norm.mean.zero_()                              # 强制 scale-only
        self.edge_norm.std.copy_(edge_std.to(self.edge_norm.std))
        self.input_stats_fitted.fill_(1.0)

    # ------------------------------------------------------------------
    # 设置加速度归一化统计量（训练前调用一次）
    # ------------------------------------------------------------------
    def set_accel_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        用训练集上计算的全局均值和标准差来初始化归一化器。

        参数：
          mean: (3,) 浮点张量 — x/y/z 三个通道的加速度均值（在整个训练集上平均）
          std:  (3,) 浮点张量 — x/y/z 三个通道的加速度标准差

        调用时机：
          训练开始前，用全部训练数据的加速度一次性计算 mean/std，然后调用此函数。
          之后训练循环中 forward() 会自动用这些统计量做 z-score 归一化。
        """
        # copy_：将 mean 的值原地拷贝到 accel_norm.mean buffer 中
        # .to() 确保设备和精度一致
        self.accel_norm.mean.copy_(mean.to(self.accel_norm.mean))

        # clamp_min(eps)：防止 std 太小导致除零（eps = 1e-6）
        # 对于退化通道（如 2D 模拟中 z 方向加速度始终 ≈ 0），
        # 我们不设很小的 std（那会放大噪声），而是设 std=1.0（保持该通道自然接近零的尺度）
        self.accel_norm.std.copy_(
            std.clamp_min(self.accel_norm.eps).to(self.accel_norm.std)
        )

        # 标记归一化器已 fit
        self.stats_fitted.fill_(1.0)

    # ------------------------------------------------------------------
    # 分块近邻搜索（no_grad, O(N·chunk) 峰值显存; 3D 大场景的关键改造）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _find_edges(self, pos, radius, factor: float, chunk: int = 4096):
        """返回无向边 (row, col), row<col, 判据 dist<factor*(r_i+r_j) 且 dist>1e-6。

        语义与旧的稠密上三角 mask 完全一致(含 row 升序的边序), 但峰值显存从
        O(N²) 降到 O(chunk·N)。搜索不参与反传(边选择本就不可微)。
        """
        N = pos.shape[0]
        rows, cols = [], []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            # 显式差平方(与旧稠密实现浮点路径逐位一致; cdist 的 mm 路径在阈值边缘会翻转)
            d2 = ((pos[s:e].unsqueeze(1) - pos.unsqueeze(0)) ** 2).sum(-1)   # (b, N)
            thr2 = (factor * (radius[s:e].unsqueeze(1) + radius.unsqueeze(0))) ** 2
            m = (d2 < thr2) & (d2 > 1e-12)
            r, c = torch.nonzero(m, as_tuple=True)
            gr = r + s
            keep = c > gr                                       # 上三角(无向唯一)
            rows.append(gr[keep]); cols.append(c[keep])
        return torch.cat(rows), torch.cat(cols)

    # ------------------------------------------------------------------
    # 构建动态接触图（每个仿真步调用一次）
    # ------------------------------------------------------------------
    def _build_graph(self, pos, vel, node_type, radius):
        """
        从粒子位置构建动态接触图。

        图是每步从头重建的——粒子移动了，邻居关系也变了。

        步骤：
          1. 计算 N×N 的距离矩阵（O(N²) 成对距离）
          2. 计算每对粒子的接触阈值 = skin_factor × (r_i + r_j)
          3. 距离 < 阈值的粒子对 → 建边（粒子感知，自动适配不同半径的粒子）
          3. 计算每条边的几何特征（相对位置、相对速度、距离）
          4. 构造节点特征向量并补齐历史维度

        参数：
          pos:       (N, 3) 浮点 — 所有粒子的三维位置
          vel:       (N, 3) 浮点 — 所有粒子的三维速度
          node_type: (N,)  长整型 — 粒子类型（0=普通粒子, 非0=边界/障碍物）
          radius:    (N,)  浮点 — 每个粒子的半径

        建边规则（粒子感知）：
          边 (i,j) 存在当且仅当：
            dist(i,j) < skin_factor × (radius[i] + radius[j])
          skin_factor = 1.25 意味着允许 25% 的预接触缓冲距离。
          这个参数不是自由超参数——它是粒子半径的确定性函数，不随工况变化。

        返回（六元组）：
          node_feats:  (N, 20) — 节点输入特征（pos+vel+radius+type+历史零填充）
          edge_index:  (2, E) — 边列表，第一行是 source，第二行是 target
          edge_attr:   (E, 7) — 边特征：rel_pos(3)+rel_vel(3)+dist(1)
          edge_ids:    (E,)   — 每条边的唯一标识符 = row*N+col，用于跨步追踪同一条边
          rel_dir:     (E, 3) — 从 col 指向 row 的单位方向向量（用于力的反对称施加）
          mass:        (N, 1) — 粒子质量（当前设为全 1，与数据一致）
        """
        # 获取 pos 张量所在的设备（CPU 或 cuda:0）
        device = pos.device

        # 粒子总数 N
        N = pos.shape[0]

        # 防御性编程：确保 radius 是 tensor
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius, dtype=torch.float32, device=device)
        if radius.ndim == 0:
            radius = radius.unsqueeze(0)
        if radius.shape[0] != N:
            radius = radius.expand(N)

        # ═══ 步骤 1+2：分块近邻搜索建边(no_grad, 见 _find_edges) ═══
        # ★3D 改造★ 旧实现物化 (N,N,3) 稠密张量且被 autograd 保存, 3D 万粒子场景 OOM;
        # 现在边选择在 no_grad 里分块完成, 边特征稀疏直接计算, 反传图中只有 O(E) 张量。
        row, col = self._find_edges(pos.detach(), radius, self.skin_factor)
        edge_index = torch.stack([row, col], dim=0)   # shape: (2, E)

        # ═══ 步骤 3a：稀疏计算每条边的几何特征(可微, 与旧稠密版逐元素一致) ═══
        rel_pos = pos[col] - pos[row]                 # = diff[row,col]
        rel_vel = vel[col] - vel[row]                 # = vel_diff[row,col]
        dist = torch.sqrt((rel_pos ** 2).sum(-1) + 1e-12)   # 与旧版同 eps

        # 拼接所有边特征：rel_pos(d) + rel_vel(d) + dist(1) = 2d+1 维
        edge_attr = torch.cat([
            rel_pos,
            rel_vel,
            dist.unsqueeze(-1)                        # (E,)→(E,1)，为拼接增加一维
        ], dim=-1)                                    # shape: (E, 2d+1)

        # ═══ 输入归一化（NEW，scale-only）═══
        # 在这里归一化一次，所有下游消费者(edge_encoder / forward 的反向边 / birth_init)
        # 都看到同一份归一化后的 edge_attr。必须在 rel_dir(下方,用 RAW 几何)之前，且因为
        # edge_norm 是 mean=0 的 scale-only，负化与归一化精确可交换 → 反向边技巧仍成立。
        if self.normalize_inputs:
            edge_attr = self.edge_norm.normalize(edge_attr)

        # ═══ 步骤 3b：计算单位方向向量（用于力的施加方向，使用 RAW 几何，不归一化） ═══
        # pos[row] - pos[col] = 从 col 指向 row 的向量（排斥方向：把 row 推离 col）
        # 除以距离 → 单位方向向量
        rel_dir = (pos[row] - pos[col]) / (dist.unsqueeze(-1) + 1e-8)
        # +1e-8 防止除以零（两个粒子完全重叠的极端情况）
        # shape: (E, 3) — 归一化的排斥方向

        # ═══ 步骤 3c：为每条边分配唯一标识符 ═══
        # 边的 ID = row*N + col（row < col 保证唯一性）
        # 只要粒子编号不变（rollout 中不变），同一条物理接触在下一步建图时会有相同的 ID
        edge_ids = row * N + col                      # shape: (E,) 长整型

        # ═══ 步骤 4：构造节点输入特征 ═══
        # radius 已在步骤 2 中验证和转换为正确形状的张量

        # 节点速度归一化（z-score）；radius(≈常数)与 type(0/1)保持 RAW
        vel_feat = self.vel_norm.normalize(vel) if self.normalize_inputs else vel

        # 拼接节点特征。两种模式：
        #   boundary_features=False: pos(d) + vel(d) + radius(1) + type(1) = 2d+2
        #   boundary_features=True : walldist(2d) + vel(d) + radius(1) + type(1) = 3d+2
        # 用"到墙裁剪距离"替换绝对位置：既给边界感知，又保持平移不变性(GNS 做法)。
        if self.boundary_features:
            assert float(self.box_size) > 0, "boundary_features 需要先设置 box_size(>0)"
            R = self.feat_clip                        # 裁剪半径 R_clip(标量 buffer)
            L = self.box_size                         # 盒边长 L(标量 buffer)
            r = radius.unsqueeze(-1)                   # (N,1) 物理墙在 coord=r / L-r 处,与 rollout clamp 对齐
            lo = (pos - r).clamp(min=0.0).clamp(max=float(R)) / R         # 到下/左/底墙距离 ∈[0,1]
            hi = (L - r - pos).clamp(min=0.0).clamp(max=float(R)) / R     # 到上/右/顶墙距离 ∈[0,1]
            node_feats = torch.cat([
                lo, hi,                               # (N, 2d) 到墙距离，替换 pos
                vel_feat,                             # (N, d)
                radius.unsqueeze(-1),                 # (N, 1) RAW(≈常数)
                node_type.unsqueeze(-1).float(),      # (N, 1) RAW(0/1)
            ], dim=-1)                                # shape: (N, 3d+2)
        else:
            node_feats = torch.cat([
                pos,                                  # (N, d) 位置(RAW)
                vel_feat,                             # (N, d) 速度(可能已归一化)
                radius.unsqueeze(-1),                 # (N, 1) 半径
                node_type.unsqueeze(-1).float(),      # (N, 1) 类型→浮点
            ], dim=-1)                                # shape: (N, 2d+2)

        # 质量：全 1（单位质量），与数据生成时的设定一致
        mass = torch.ones(N, 1, device=device)        # shape: (N, 1)

        return node_feats, edge_index, edge_attr, edge_ids, rel_dir, mass

    # ------------------------------------------------------------------
    # 边记忆的跨步传递：新接触→初始化，旧接触→继承上一帧的记忆
    # ------------------------------------------------------------------
    def _carry_memory(self, edge_attr, edge_ids, edge_memory_state, edge_id_map):
        """
        处理边记忆的"诞生"和"继承"。

        这是 Trace 实现"跨步持久记忆"的核心工程函数。
        每一步建图后调用：
          - 上一步存在的边 → 从 edge_memory_state 中取出旧记忆，继承到当前步
          - 上一步不存在的边（新接触）→ 调用 edge_memory.initialize() 初始化新记忆

        使用 torch.where（不在 autograd 张量上做原地操作），保证 BPTT 安全。

        参数：
          edge_attr:         (E, 7)  浮点 — 当前步所有边的特征
          edge_ids:          (E,)   长整型 — 当前步每条边的唯一标识符
          edge_memory_state: (E_prev, carry_dim) — 上一步携带过来的边记忆状态
          edge_id_map:       dict{边ID→索引} — 上一步的边 ID 到数组索引的映射

        返回：
          carried:    (E, carry_dim) — 当前步每条边的记忆（继承或新初始化）
          new_id_map: dict{边ID→索引} — 当前步的边 ID 映射（传给下一步使用）
        """
        # 当前步的边数 E
        E = edge_attr.shape[0]

        # 极端情况：没有边（所有粒子相距太远）→ 返回空记忆和空映射
        if E == 0:
            # new_zeros 创建一个与 edge_attr 在同设备/同精度的全零张量
            return edge_attr.new_zeros((0, self.edge_memory.carry_dim)), {}

        # ── 先为所有边生成"新接触"的初始记忆 ──
        # initialize() 通过 birth_init MLP 从 7 维边特征映射到 carry_dim 维初始记忆
        # 注意：这是有梯度的（MLP 参数可训练），所以即使后续被覆盖，梯度也能流回
        init_memory = self.edge_memory.initialize(edge_attr)  # shape: (E, carry_dim)

        # ── 构建当前步的边 ID → 索引 映射 ──
        eids = edge_ids.tolist()                              # 转为 Python 列表（用于构建 dict）
        new_id_map = {eid: i for i, eid in enumerate(eids)}   # {边ID: 在数组中的位置}

        # ── 如果上一步有记忆，尝试继承 ──
        if (edge_memory_state is not None            # 上一步的记忆存在
            and edge_id_map is not None               # 上一步的 ID 映射存在
            and edge_memory_state.shape[0] > 0):      # 上一步至少有一条边

            # 对当前步的每条边，查找它在上一步的索引
            # edge_id_map.get(eid, -1)：
            #   如果这条边在上一步存在 → 返回其索引
            #   否则 → 返回 -1（表示这是新接触）
            prev_idx = torch.tensor(
                [edge_id_map.get(eid, -1) for eid in eids],
                device=edge_attr.device,              # 与 edge_attr 在同一设备
                dtype=torch.long,                     # 整数索引类型
            )                                         # shape: (E,)

            # 标记哪些边是"旧接触"（prev_idx >= 0 表示在上一步存在）
            valid = (prev_idx >= 0).unsqueeze(-1)     # shape: (E, 1) 布尔型

            # 从旧记忆中取出对应边
            # 对 prev_idx=-1 的新接触，用 clamp(min=0) 临时取第 0 条（后面会被 torch.where 替换掉）
            gathered = edge_memory_state[prev_idx.clamp(min=0)]  # shape: (E, carry_dim)

            # torch.where(condition, x, y)：
            #   条件为 True (valid) 的位置 → 取 x（继承的旧记忆）
            #   条件为 False         的位置 → 取 y（新初始化的记忆，有梯度）
            # 关键：不修改原张量，保持 autograd 图的完整性
            carried = torch.where(valid, gathered, init_memory)
        else:
            # 第一步（或上一步没有边）：所有边都视为新接触
            carried = init_memory

        return carried, new_id_map

    # ------------------------------------------------------------------
    # 前向传播：一个仿真步的完整计算
    # ------------------------------------------------------------------
    def forward(
        self,
        pos: torch.Tensor,                              # (N, 3) 浮点 — 当前步的粒子位置
        vel: torch.Tensor,                              # (N, 3) 浮点 — 当前步的粒子速度
        node_type: torch.Tensor,                        # (N,)  长整型 — 粒子类型（0=普通）
        radius: torch.Tensor,                           # (N,)  浮点 — 粒子半径
        material_id: Optional[torch.Tensor] = None,     # (N,)  长整型 — 材料 ID（可选，当前未使用）
        edge_memory_state: Optional[torch.Tensor] = None,  # (E_prev,carry_dim) — 上一步的边记忆
        edge_id_map: Optional[Dict[int, int]] = None,   # 上一步的边 ID 映射
        training_noise: bool = True,                    # 训练时 True（加噪声）；推理时 False
    ) -> Dict[str, torch.Tensor]:
        """
        一个仿真步的前向传播。这是模型训练和推理的入口。

        返回字典包含：
          - accel:            (N,3) 归一化的加速度 → 用于计算 loss
          - accel_phys:       (N,3) 物理加速度 → 用于积分更新粒子状态
          - internal_accel:   (N,3) 接触力贡献的加速度 → 用于分析和可视化
          - F_n/F_t/mu:       每条边的法向力、切向力、学习到的摩擦系数
          - edge_memory_state: (E,carry_dim) 携带到下一步的边记忆
          - edge_id_map:      当前步的边 ID 映射
        """
        # ═══ 步骤 1：构建动态接触图 ═══
        node_feats, edge_index, edge_attr, edge_ids, rel_dir, mass = self._build_graph(
            pos, vel, node_type, radius
        )
        # node_feats: (N,node_in_dim)  edge_index: (2,E)  edge_attr: (E,2d+1)
        # edge_ids:   (E,)             rel_dir:    (E,d)   mass:      (N,1)

        # 防御：开了输入归一化但没 fit 统计量 = 静默无操作(会复现欠拟合 bug)，训练时直接报错
        if self.normalize_inputs and self.training:
            assert float(self.input_stats_fitted) > 0, \
                "normalize_inputs=True 但输入统计量未拟合(需调用 set_input_stats)"

        # 拆出源节点索引和目标节点索引
        row, col = edge_index[0], edge_index[1]         # 各 (E,) 长整型
        E = edge_index.shape[1]                         # 边数 E

        # ═══ 步骤 2：编码 → 隐含空间 ═══
        # 节点编码：(N,20) → (N,128)
        node_h = self.node_encoder(node_feats)

        # 边编码：(E,7) → (E,128)
        # 如果没有边（E=0），创建形状为 (0,128) 的空张量，保持计算一致性
        if E > 0:
            edge_h = self.edge_encoder(edge_attr)       # (E, 128)
        else:
            # new_zeros：创建一个与 node_h 在同一设备/精度的全零张量
            edge_h = node_h.new_zeros((0, self.hidden_dim))  # (0, 128)

        # ═══ 步骤 3：边记忆的跨步传递 + 更新（Trace 的核心创新） ═══
        # 3a. 携带/诞生记忆：旧接触继承，新接触初始化
        carried, new_id_map = self._carry_memory(
            edge_attr, edge_ids, edge_memory_state, edge_id_map
        )
        # carried: (E, carry_dim) — 当前步每条边的"携带状态"（继承或新初始化）
        # 注意 carry_dim 可能 ≠ memory_dim（对于 st2 等变体）

        # 3b. 更新记忆（GRU 门控 / 注意力）——每步只更新一次！
        if E > 0:
            # edge_memory.update() 是多态接口，根据 memory_type 调度到不同实现：
            #   EdgeMemoryState(GPU)：
            #     → carried 过 GRU 门控更新，返回 (E, memory_dim)
            #   EdgeSpatioTemporalMemory(st/stmh)：
            #     → carried 先过空间线图注意力（边看到邻居边），再过 GRU 更新
            #   EdgeSpatioTemporalAttn(st2)：
            #     → 纯注意力更新（窗口化时间注意力 + 空间注意力）
            carry = self.edge_memory.update(
                carried,                                # (E, carry_dim) 继承状态
                edge_h,                                 # (E, 128) 当前步编码后的边特征
                edge_index=edge_index                   # (2, E) 用于空间注意力的邻接信息
            )
            # carry: (E, carry_dim) — 更新后的携带状态（将传给下一步）
        else:
            carry = carried                              # 没有边时直接保留

        # 3c. 提取"当前记忆"用于处理器和解码器
        # current() 将 carry 状态映射回 memory_dim 维（对于 GRU 版本是恒等映射，
        # 对于 st2 版本是从窗口化记忆中取最后一帧）
        edge_mem = self.edge_memory.current(carry)       # (E, memory_dim) = (E, 16)

        # ═══ 步骤 4：处理器 — K 层感知边记忆的消息传递 ═══
        # 边记忆 edge_mem 在本步内是固定的（已在步骤 3 中更新过）
        # 消息传递只读取边记忆，不修改它

        # ── 构建双向边用于消息传递 ──
        # 构图时每条物理接触只保留一条无向边 (row<col)，但消息传递需要两个方向
        # 反向边：rel_pos/rel_vel 取反，距离不变，共享同一条边记忆
        if E > 0:
            # 反向边：rel_pos/rel_vel 取反，dist(标量,对称)不变。
            # ★ dim-generic：edge_attr 宽度=2d+1，前 2d 维是方向量(取反)、最后 1 维是 dist。
            #   旧代码硬编码 [:3,3:6,6:7] 仅对 3D 正确，2D(宽度5)会把 dist 当 rel_vel 取反。
            #   因 edge_attr 已是 scale-only(mean=0)归一化，-rel == normalize(-rel_raw) 精确成立。
            d = self.dim
            rel = edge_attr[:, :2 * d]                 # rel_pos(d)+rel_vel(d) 方向量
            dst = edge_attr[:, 2 * d:2 * d + 1]        # dist(1) 对称
            edge_attr_rev = torch.cat([-rel, dst], dim=-1)   # (E, 2d+1)
            edge_h_bi = torch.cat([
                edge_h,
                self.edge_encoder(edge_attr_rev)
            ], dim=0)                                  # (2E, 128)
            edge_mem_bi = edge_mem.repeat(2, 1)        # (2E, 16) 共享记忆

            edge_index_bi = torch.stack([
                torch.cat([row, col]),                 # 正向：col→row
                torch.cat([col, row]),                 # 反向：row→col
            ], dim=0)                                  # (2, 2E)
        else:
            edge_index_bi = edge_index
            edge_h_bi = edge_h
            edge_mem_bi = edge_mem

        for blk, norm in zip(self.processor, self.processor_norms):
            # blk: MemoryEdgeBlock — 边记忆感知的消息传递层
            # norm: RMSNorm — 归一化层
            node_h, _ = blk(
                node_h,             # (N, 128) 当前节点隐含状态
                edge_index_bi,      # (2, 2E) 双向图结构
                edge_h_bi,          # (2E,128) 双向边隐含特征
                edge_mem_bi         # (2E, 16) 双向边记忆（正反向共享）
            )
            # 返回：更新后的 node_h(N,128) 和 edge_msg(2E,128)
            # 我们不使用 edge_msg，仅 node_h 继续传递

        # ═══ 步骤 5：解码 — 隐含状态 → 物理量 ═══
        # 5a. 外部加速度（每个粒子独立学习）：
        #     隐含状态(N,128) → Linear→SiLU→LayerNorm→Linear → 外部加速度(N,3)
        ext_accel = self.node_decoder(node_h)

        # 5b. 内部接触力（成对）：
        #     边记忆(E,16) + 边隐含(E,128) → MLP → F_n(E,1), F_t(E,3), μ(E,1)
        if E > 0:
            F_n, F_t, mu = self.force_decoder(edge_mem, edge_h)
            # F_n: (E,1) — 法向力大小（≥0，softplus 保证）
            # F_t: (E,3) — 切向力向量（已被 Coulomb 锥夹紧：|F_t| ≤ μ·F_n）
            # mu:  (E,1) — 学习到的摩擦系数 ∈ [0.1, 1.0]

            # ═══ 步骤 6：动量守恒 — Newton's Third Law ═══
            # f_edge = F_n·n + F_t：
            #   F_n·n 是沿排斥方向的法向力（n 从 col 指向 row，即"把 row 推离 col"）
            #   F_t 是切向摩擦力
            f_edge = F_n * rel_dir + F_t               # (E, 3)：每条边产生的总接触力向量

            # net 初始化为零：(N,3)
            net = torch.zeros_like(ext_accel)

            # index_add_(dim, index, source)：
            #   将 source 按 index 指定的位置加到 net 的 dim 维上
            #
            # net[row] += f_edge
            # → 力 +f_edge 施加到源节点 row（把 row 推开的方向）
            net.index_add_(0, row, f_edge)

            # net[col] += -f_edge
            # → 力 -f_edge 施加到目标节点 col（大小相等，方向相反）
            # → 同一条边的两个粒子受力等大反向
            # → 系统总内力恒为零 → 线性动量守恒！
            net.index_add_(0, col, -f_edge)

            # 除以粒子质量得到加速度贡献（当前质量全为 1.0）
            internal_accel = net / mass                # (N, 3)
        else:
            # 没有边时：无内部力，创建空张量以保持返回格式一致
            F_n = pos.new_zeros((0, 1))                # new_zeros 创建与 pos 同设备/精度的空张量
            F_t = pos.new_zeros((0, self.dim))         # 切向力维度随 d
            mu = pos.new_zeros((0, 1))
            internal_accel = torch.zeros_like(ext_accel)  # 全零 (N,3)

        # ═══ 步骤 7：组装总加速度 ═══
        # 总加速度 = 外部（重力+边界力）+ 内部（接触力）
        accel_phys = ext_accel + internal_accel        # (N, 3) — 物理加速度，用于积分

        # z-score 归一化：(accel_phys - mean) / std
        # 归一化后的加速度用于计算 loss（在归一化空间中 loss 更稳定）
        accel = self.accel_norm.normalize(accel_phys)  # (N, 3) — 归一化加速度，用于 loss

        # ═══ 步骤 8：返回所有结果 ═══
        return {
            "accel": accel,                             # (N,3) 归一化加速度 → 用于 loss 计算
            "accel_phys": accel_phys,                   # (N,3) 物理加速度 → 用于半隐式 Euler 积分
            "internal_accel": internal_accel,           # (N,3) 接触力贡献 → 用于分析和可视化
            "F_n": F_n,                                 # (E,1) 每条边的法向力 → 用于物理指标验证
            "F_t": F_t,                                 # (E,3) 每条边的切向力 → 用于物理指标验证
            "mu": mu,                                   # (E,1) 学习到的摩擦系数
            "edge_memory_state": carry,                 # (E,carry_dim) 携带到下一步的边记忆状态
            "edge_id_map": new_id_map,                  # dict{边ID→索引} 传给下一步的 ID 映射
        }

    # ------------------------------------------------------------------
    # 不可穿透约束求解器（推理时使用，训练时不调用）
    # ------------------------------------------------------------------
    @torch.no_grad()                                    # 禁用梯度计算——这纯粹是约束投影，不涉及学习
    def _resolve_overlaps(self, pos, vel, radius, dt, iters=25):
        """
        基于位置的不可穿透投影（Position-Based Dynamics 风格的约束求解）。

        为什么需要这个？
        纯学习的 GNN 模拟器在刚性接触下不可避免地会出现粒子重叠
        （particles interpenetrate / 互相"煎饼"）。这不是 Trace 的问题，
        而是所有无约束学习模拟器的通病。

        这个函数充当"接触约束层"——在 GNN 预测加速度并更新位置后，
        迭代地将重叠的粒子对沿接触法向推开。这不是学习方法，而是
        模拟器层面的硬约束——物理上，刚性球不能穿透彼此。

        算法：Jacobi 迭代（并行处理所有粒子对）
          for iter in range(iters):
              对所有粒子对 (i,j)，如果重叠(overlap > 0)：
                dvec = pos[i] - pos[j]          ← 从 j 指向 i 的方向
                沿 dvec/|dvec| 把 i 和 j 各推开 overlap/2

        参数：
          pos:    (N,3) — 当前粒子位置
          vel:    (N,3) — 当前粒子速度（本函数不修改速度，只返回原值）
          radius: (N,)  — 粒子半径
          dt:     float — 时间步长（保留以备将来使用，当前未用）
          iters:  int   — Jacobi 迭代次数（默认 25，越多越精确但越慢）

        返回：
          pos: (N,3) — 修正后的粒子位置（无穿透）
          vel: (N,3) — 原样返回的速度
        """
        N = pos.shape[0]                                # 粒子数 N
        if N < 2:
            return pos, vel

        # ★3D 改造★ 旧实现每轮迭代物化 (N,N,3) 张量(万粒子场景单轮 ~1-5GB × 25 轮),
        # 改为: 一次分块搜索找出候选接触对(含 50% 裕量, 覆盖迭代中的微小位移),
        # 之后 25 轮 Jacobi 只在候选对上计算并用 index_add 聚合——语义与稠密版一致。
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius, dtype=pos.dtype, device=pos.device)
        if radius.ndim == 0:
            radius = radius.expand(N)
        row = col = rsum = None
        for it in range(iters):
            if it % 5 == 0:                             # 候选对每 5 轮刷新(防大位移逃出裕量)
                row, col = self._find_edges(pos, radius, factor=2.0)
                if row.numel() == 0:
                    return pos, vel
                rsum = radius[row] + radius[col]        # (P,) 每对的最小不穿透距离
            dvec = pos[row] - pos[col]                  # (P,3) 从 col 指向 row
            dist = dvec.norm(dim=-1)
            ov = (rsum - dist).clamp(min=0.0)           # (P,) 正值=穿透量
            nrm = dvec / (dist.unsqueeze(-1) + 1e-9)
            push = 0.5 * ov.unsqueeze(-1) * nrm         # 每对各推开一半
            corr = torch.zeros_like(pos)
            corr.index_add_(0, row, push)
            corr.index_add_(0, col, -push)
            pos = pos + corr

        return pos, vel                                # 返回修正后的位置和原速度

    # ------------------------------------------------------------------
    # 一步积分后的硬约束（推理 rollout 与训练漂移/微调共用 → 训推一致，P2）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _constrain_state(self, pos, vel, prev_pos, radius, box_size, dt,
                         contact_projection: bool = True, proj_iters: int = 25):
        """对积分后的 (pos, vel) 施加与推理完全相同的硬约束。

        顺序与 rollout() 的旧实现一致：
          1. 粒子间不可穿透 Jacobi 投影（可关）
          2. 地板 / 侧墙 clamp（位置贴墙 + 法向速度截断）
          3. （可选，P3）vel_from_displacement：用位移重算速度
             v = (pos_constrained - prev_pos) / dt，使模型输入恢复数据约定
             vel[t] = pos[t] - pos[t-1]（投影/clamp 移动了 pos 之后，积分速度
             与位移速度会失配，长时下持续累积分布偏移）。

        注意：会原地修改传入的 pos/vel（调用方传新建张量或先 clone）。
        """
        L = box_size
        if contact_projection:
            pos, vel = self._resolve_overlaps(pos, vel, radius, dt, proj_iters)

        # 地板（y 轴 = 维度 1）
        below = pos[:, 1] < radius
        pos[below, 1] = radius[below]
        vel[below, 1] = vel[below, 1].clamp(min=0)

        # 侧墙：除"上"轴(维度1)外的所有水平轴
        for ax in [a for a in range(self.dim) if a != 1]:
            lo = pos[:, ax] < radius
            pos[lo, ax] = radius[lo]
            vel[lo, ax] = vel[lo, ax].clamp(min=0)
            hi = pos[:, ax] > L - radius
            pos[hi, ax] = L - radius[hi]
            vel[hi, ax] = vel[hi, ax].clamp(max=0)

        if self.vel_from_displacement:
            vel = (pos - prev_pos) / dt

        return pos, vel

    # ------------------------------------------------------------------
    # 自回归 Rollout（推理/可视化用）
    # ------------------------------------------------------------------
    def rollout(
        self,
        pos_0: torch.Tensor,                            # (N,3) 初始粒子位置
        vel_0: torch.Tensor,                            # (N,3) 初始粒子速度
        node_type: torch.Tensor,                        # (N,)  粒子类型
        radius: torch.Tensor,                           # (N,)  粒子半径
        n_steps: int,                                   # 要模拟的步数（例如 200）
        dt: float = 0.001,                              # 时间步长（秒）
        box_size: float = 1.0,                          # 盒子边长（用于边界约束）
        contact_projection: bool = True,                # 是否启用不可穿透投影
        proj_iters: int = 25,                           # 投影的 Jacobi 迭代次数
    ) -> List[Dict[str, torch.Tensor]]:
        """
        自回归 Rollout：从初始状态出发，反复调用 forward() 推进仿真。

        积分方案：半隐式 Euler（比显式 Euler 更稳定）
          v_{t+1} = v_t + a_t * dt          (1) 先用旧加速度更新速度
          x_{t+1} = x_t + v_{t+1} * dt      (2) 再用新速度更新位置

        两个硬约束（每步执行，不属于学习动力学）：
          1. 盒边界：粒子不能穿透地板（y=radius）和四面墙壁（x,z = [radius, L-radius]）
          2. 粒子间不可穿透：基于位置的 Jacobi 投影（_resolve_overlaps）

        参数：
          pos_0:              (N,3) 初始位置
          vel_0:              (N,3) 初始速度
          node_type:          (N,)  粒子类型
          radius:             (N,)  粒子半径
          n_steps:            int   模拟步数
          dt:                 float 时间步长
          box_size:           float 盒子边长
          contact_projection: bool  是否启用不可穿透投影（默认 True）
          proj_iters:         int   投影迭代次数（越大越精确）

        返回：
          轨迹列表，每个元素是一个字典，包含该帧的 pos, vel, accel, F_n, F_t 等
        """
        # ── 初始化状态（深拷贝，不修改输入）──
        pos = pos_0.clone()                             # .clone() 创建独立副本
        vel = vel_0.clone()

        # 边界特征与下方的边界 clamp 必须用同一个 box_size：把 rollout 的 box_size 参数
        # 钉进 buffer，保证 _build_graph 里的到墙距离和 clamp 不会用到不同的 L。
        if self.boundary_features:
            self.box_size.fill_(float(box_size))

        # 防御性编程：确保 radius 是正确形状的 tensor
        if not isinstance(radius, torch.Tensor):
            radius = torch.as_tensor(radius, dtype=pos.dtype, device=pos.device)
            # as_tensor 不复制数据（如果已经是 tensor），保持设备一致
        if radius.ndim == 0:                            # 标量 → 扩展为每粒子
            radius = radius.expand(pos.shape[0])

        L = box_size                                    # 盒子边长（简化变量名）

        # 第一帧没有任何上一帧的记忆，所以都是 None
        memory, id_map = None, None

        # 轨迹存储列表，每个元素是 rollout 中一帧的完整状态
        trajectory = []

        # ── 主循环：逐帧推进 ──
        for _ in range(n_steps):
            # 调用 forward 预测当前步的加速度
            # training_noise=False：推理时不加噪声（噪声只在训练时使用）
            result = self.forward(
                pos, vel, node_type, radius,
                edge_memory_state=memory,               # 传入上一步的边记忆
                edge_id_map=id_map,                     # 传入上一步的 ID 映射
                training_noise=False,                   # 推理模式：不加噪声！
            )

            # 提取物理加速度（denormalized，即归一化前的原始值）
            # 积分必须用物理空间的值，不能用归一化空间的值
            a_phys = result["accel_phys"]               # (N, 3)

            # ── 半隐式 Euler 积分 ──
            prev_pos = pos                              # 约束前的旧位置（P3 位移速度要用）
            # 步骤 (1)：用当前加速度更新速度
            vel = vel + a_phys * dt                     # v_new = v_old + a * dt
            # 步骤 (2)：用新速度更新位置
            pos = pos + vel * dt                        # x_new = x_old + v_new * dt
            # 注意：用 v_new 而非 v_old！这就是"半隐式"的含义——比显式 Euler 更稳定

            # ═══ 硬约束（穿透投影 + 盒边界，训推共用 _constrain_state）═══
            pos, vel = self._constrain_state(pos, vel, prev_pos, radius, L, dt,
                                             contact_projection=contact_projection,
                                             proj_iters=proj_iters)

            # ── 保存当前帧的边记忆和 ID 映射，供下一帧使用 ──
            memory = result["edge_memory_state"]        # (E, carry_dim) 携带状态
            id_map = result["edge_id_map"]              # dict{边ID→索引}

            # ── 记录当前帧的完整状态（深拷贝以避免被后续修改）──
            trajectory.append({
                "pos": pos.clone(),                     # (N,3) 当前位置
                "vel": vel.clone(),                     # (N,3) 当前速度
                "accel": a_phys.clone(),                # (N,3) 总物理加速度
                "internal_accel": result["internal_accel"].clone(),  # (N,3) 接触力贡献
                "F_n": result["F_n"].clone(),           # (E,1) 每条边的法向力
                "F_t": result["F_t"].clone(),           # (E,3) 每条边的切向力
            })

        return trajectory


# ═══════════════════════════════════════════════════════════════════════════════
# 消融实验变体
# ═══════════════════════════════════════════════════════════════════════════════

class Trace_NoMemory(Trace):
    """
    消融变体 1：无记忆 Trace（对应 Block A 实验 A4）。

    通过强制禁用边记忆的跨步传递，测试"边记忆是否真的在起作用"。
    如果这个变体的性能和主模型一样 → 边记忆没有贡献 → 核心假设不成立。

    实现方式：在调用 forward 之前强制将 edge_memory_state 和 edge_id_map 设为 None，
    使得每一步的每条边都被当作"新接触"重新初始化（birth_init）。

    使用方式：
      model = Trace_NoMemory(hidden_dim=128, ...)
      # 其他接口完全同 Trace
    """

    def forward(self, *args, **kwargs):
        # 强制禁用跨步记忆传递
        kwargs["edge_memory_state"] = None              # 不传入上一步的记忆
        kwargs["edge_id_map"] = None                    # 不传入上一步的 ID 映射
        # 调用父类 Trace.forward：
        #   _carry_memory 发现 edge_memory_state=None → 所有边都用 init_memory（新初始化）
        #   效果等价于无跨步记忆
        return super().forward(*args, **kwargs)


class Trace_Supervised(Trace):
    """
    消融变体 2：监督记忆 Trace（对应 Block A 实验 A2）。

    在主模型的涌现记忆训练基础上，额外监督切向接触力 F_t。
    用于测量"从宏观轨迹自发学会摩擦历史"和"直接告诉模型正确答案"之间的差距。

    如果 A1（涌现）需要 A2（监督）才能通过 Block A 门控实验 → C3 失败 → 转向 Idea B。

    使用方式：
      model = Trace_Supervised(hidden_dim=128, ...)
      # 训练时在 loss 中加入切向力监督项：
      #   tangent_loss = model.tangent_loss(pred["F_t"], target["F_t"])
      #   total_loss = main_loss + model.tangent_loss_weight * tangent_loss
    """

    def __init__(self, **kwargs):
        # **kwargs 将所有参数透明传递给 Trace.__init__
        super().__init__(**kwargs)
        # 切向力损失在总 loss 中的权重（可调，默认 1.0）
        self.tangent_loss_weight = 1.0

    def tangent_loss(self, F_t_pred: torch.Tensor, F_t_true: torch.Tensor) -> torch.Tensor:
        """
        计算预测切向力和真实 DEM 切向力之间的均方误差。

        注意：在 Trace 主模型（涌现记忆）中，此函数从不被调用！
        仅在消融实验 A2 中使用，用于测量"omniscient supervision"的上界。

        参数：
          F_t_pred: (E, 3) 浮点 — Trace 预测的每条边的切向力向量
          F_t_true: (E, 3) 浮点 — DEM 求解器记录的每条边的真实切向力向量

        返回：
          标量 MSE loss（越小越好）
        """
        # torch.nn.functional.mse_loss：逐元素均方误差，自动求平均
        return torch.nn.functional.mse_loss(F_t_pred, F_t_true)
