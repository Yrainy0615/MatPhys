# Zero-Shot Material Physics Training Design

> **Implementation**: See `models.py` and `train_zeroshot.py`

## 1. 问题分析

### 当前状态
- **DINO特征**: `{case}_node_sem.npz` → `[num_nodes, 1024]` (per-node semantic features)
- **Part Clustering**: K-means on DINO features → `[K parts]` per case
- **Part Features**: 聚合后的part-level DINO特征 → `[K, 1024]`
- **Material Embedding**: 随机初始化 `[K, 64]`，未与DINO space关联

### 核心问题
1. **Material Embedding与DINO特征断裂**: 当前embedding是随机的，没有利用DINO的语义信息
2. **Zero-shot能力缺失**: 新case必须重新训练，无法利用DINO特征直接推理
3. **Prior太小**: 当前用5个类别 × 16维，表达能力不足

### 目标
- **Zero-shot推理**: `DINO feature → material embedding → physics parameter`
- **样本高效**: 少量样本即可学习，不需要VAE等复杂结构
- **可解释**: 中间的part/material分布可以可视化验证

---

## 2. 核心设计思想

### 关键洞察
DINO特征本身已经编码了丰富的材料语义信息：
- 相似材料的DINO特征在高维空间中聚类
- 通过clustering，我们已经得到了part-level的语义分组
- 每个part的DINO特征（聚合后）可以作为该part的**材料先验**

### 设计原则
1. **DINO作为Anchor**: 用DINO特征空间作为材料语义的anchor
2. **学习投影而非embedding**: 学习从DINO space到Physics space的投影
3. **Part-level而非Node-level**: 在part级别学习，降低复杂度
4. **共享投影网络**: 所有case共享同一个投影网络，实现泛化

---

## 3. 训练架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Zero-Shot Material Physics Architecture               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: New Case                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  1. Extract DINO features: z_node [N, 1024]                      │   │
│  │  2. K-means clustering → part_assignments [N], K parts           │   │
│  │  3. Aggregate per-part: z_part [K, 1024]                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  Material Projection Network                      │   │
│  │                     (Shared, Learnable)                           │   │
│  │                                                                   │   │
│  │   z_part [K, 1024] ──► ProjNet ──► z_material [K, 128]           │   │
│  │                                                                   │   │
│  │   ProjNet = Linear(1024, 512) → LN → SiLU                        │   │
│  │           → Linear(512, 256) → LN → SiLU                         │   │
│  │           → Linear(256, 128)                                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Broadcast to Edge Level                              │   │
│  │                                                                   │   │
│  │   For edge (i,j):                                                 │   │
│  │     part_i, part_j = part_assignments[i], part_assignments[j]    │   │
│  │     z_edge = (z_material[part_i] + z_material[part_j]) / 2       │   │
│  │                                                                   │   │
│  │   Result: z_edge [num_edges, 128]                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Physics Decoder                                │   │
│  │                     (Shared, Learnable)                           │   │
│  │                                                                   │   │
│  │   Input: [z_edge(128) + z_geo(10) + edge_mid(3)] = 141-dim       │   │
│  │                                                                   │   │
│  │   Decoder = Linear(141, 256) → SiLU                              │   │
│  │           → Linear(256, 256) → SiLU                              │   │
│  │           → Linear(256, 1)  # log(k)                             │   │
│  │                                                                   │   │
│  │   Output: log_k [num_edges, 1]                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Physics Simulation                             │   │
│  │                                                                   │   │
│  │   k = exp(log_k) → Inject into mass-spring system                │   │
│  │   Simulate → L_track + L_geo + L_render                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 模块设计

### 4.1 MaterialProjectionNet

将DINO特征投影到物理材料空间：

```python
class MaterialProjectionNet(nn.Module):
    """
    Project DINO features to physics material space.

    This is the KEY learnable component that bridges
    visual semantics (DINO) and physical properties.
    """
    def __init__(
        self,
        dino_dim: int = 1024,
        material_dim: int = 128,
        hidden_dims: List[int] = [512, 256],
    ):
        super().__init__()

        dims = [dino_dim] + hidden_dims + [material_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No LN/activation on last layer
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

    def forward(self, z_dino: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_dino: [K, 1024] part-level DINO features
        Returns:
            z_material: [K, 128] material embeddings
        """
        return self.net(z_dino)
```

### 4.2 PhysicsDecoder

从材料embedding预测物理参数：

```python
class PhysicsDecoder(nn.Module):
    """
    Decode material embedding + geometry to physics parameters.
    """
    def __init__(
        self,
        material_dim: int = 128,
        geo_dim: int = 10,
        coord_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()

        input_dim = material_dim + geo_dim + coord_dim  # 141

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # log(k)
        )

    def forward(
        self,
        z_material: torch.Tensor,  # [num_edges, 128]
        z_geo: torch.Tensor,       # [num_edges, 10]
        edge_mid: torch.Tensor,    # [num_edges, 3]
    ) -> torch.Tensor:
        """
        Returns:
            log_k: [num_edges, 1] predicted log stiffness
        """
        h = torch.cat([z_material, z_geo, edge_mid], dim=-1)
        return self.net(h)
```

### 4.3 ZeroShotMaterialPhysics (Complete Model)

```python
class ZeroShotMaterialPhysics(nn.Module):
    """
    Complete model for zero-shot material physics prediction.

    Pipeline:
        DINO part features → Material Projection → Physics Decoder → k
    """
    def __init__(
        self,
        dino_dim: int = 1024,
        material_dim: int = 128,
        geo_dim: int = 10,
        coord_dim: int = 3,
        proj_hidden: List[int] = [512, 256],
        decoder_hidden: int = 256,
    ):
        super().__init__()

        self.proj_net = MaterialProjectionNet(
            dino_dim=dino_dim,
            material_dim=material_dim,
            hidden_dims=proj_hidden,
        )

        self.decoder = PhysicsDecoder(
            material_dim=material_dim,
            geo_dim=geo_dim,
            coord_dim=coord_dim,
            hidden_dim=decoder_hidden,
        )

    def forward(
        self,
        z_part: torch.Tensor,           # [K, 1024] part-level DINO
        part_assignments: torch.Tensor, # [num_edges, 2] edge endpoint parts
        z_geo: torch.Tensor,            # [num_edges, 10]
        edge_mid: torch.Tensor,         # [num_edges, 3]
    ) -> torch.Tensor:
        """
        Args:
            z_part: Part-level DINO features
            part_assignments: Part indices for each edge's endpoints
            z_geo: Geometric features per edge
            edge_mid: Edge midpoint coordinates

        Returns:
            log_k: [num_edges, 1] predicted log stiffness
        """
        # Project DINO → material space
        z_material_parts = self.proj_net(z_part)  # [K, 128]

        # Broadcast to edges (average of endpoint parts)
        part_i = part_assignments[:, 0]  # [num_edges]
        part_j = part_assignments[:, 1]  # [num_edges]
        z_edge = (z_material_parts[part_i] + z_material_parts[part_j]) / 2

        # Decode to physics
        log_k = self.decoder(z_edge, z_geo, edge_mid)

        return log_k
```

---

## 5. 数据准备

### 5.1 Per-Case数据结构

```python
@dataclass
class CaseData:
    case_name: str

    # Node-level
    node_positions: Tensor      # [N, 3]
    node_dino: Tensor           # [N, 1024] from {case}_node_sem.npz

    # Part-level (from clustering)
    part_assignments: Tensor    # [N] node → part mapping
    num_parts: int              # K
    part_features: Tensor       # [K, 1024] aggregated DINO per part

    # Edge-level
    edge_indices: Tensor        # [E, 2] edge endpoint node indices
    edge_part_indices: Tensor   # [E, 2] edge endpoint part indices
    z_geo: Tensor               # [E, 10] geometric features
    edge_mid: Tensor            # [E, 3] edge midpoints

    # Ground truth (from first-order optimization)
    teacher_logk: Tensor        # [E, 1] ground truth log(k)
```

### 5.2 数据加载流程

```python
def prepare_case_data(case_name: str) -> CaseData:
    """
    1. Load node DINO features from cache
    2. Run K-means clustering (or load cached)
    3. Aggregate to part-level features
    4. Build edge topology and compute z_geo
    5. Load teacher_logk from optimization checkpoint
    """
    # Load DINO features
    node_dino = np.load(f"semantic/cache/{case_name}_node_sem.npz")["node_sem"]

    # Clustering (cached in train_ready.pt if available)
    train_ready = torch.load(f"results/{case_name}/train/train_ready.pt")
    part_assignments = train_ready["part_assignments"]
    part_features = train_ready["part_features"]

    # Build edge topology (from material_param_dataset.py)
    ...

    return CaseData(...)
```

---

## 6. 训练策略

### 6.1 两阶段训练

**Stage 1: Teacher Supervision (Warm-up)**
- 用first-order optimization的结果作为teacher
- 快速学习DINO → k的基本映射
- Loss: `L1(pred_logk, teacher_logk)`

```python
def train_stage1(model, dataloader, epochs=100):
    """Teacher supervision stage."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch in dataloader:
            pred_logk = model(
                batch["part_features"],
                batch["edge_part_indices"],
                batch["z_geo"],
                batch["edge_mid"],
            )
            loss = F.l1_loss(pred_logk, batch["teacher_logk"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**Stage 2: Physics Refinement (Optional)**
- 用物理模拟loss微调
- 确保预测的k在模拟中表现良好
- Loss: `L_track + L_geo + λ_render * L_render`

```python
def train_stage2(model, case_runtimes, epochs=50):
    """Physics simulation refinement."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for case_name, runtime in case_runtimes.items():
            # Forward
            pred_logk = model(...)

            # Inject into simulator
            runtime.sim.inject_spring_stiffness(pred_logk.exp())

            # Simulate and compute physics loss
            L_physics = runtime.simulate_and_compute_loss()

            optimizer.zero_grad()
            L_physics.backward()
            optimizer.step()
```

### 6.2 正则化

```python
# 1. DINO space consistency: 相似DINO特征 → 相似material embedding
def contrastive_loss(z_material, z_dino, temperature=0.1):
    """
    Encourage similar DINO features to have similar material embeddings.
    """
    # Cosine similarity in DINO space
    dino_sim = F.cosine_similarity(z_dino.unsqueeze(0), z_dino.unsqueeze(1), dim=-1)

    # Cosine similarity in material space
    mat_sim = F.cosine_similarity(z_material.unsqueeze(0), z_material.unsqueeze(1), dim=-1)

    # Alignment loss
    loss = F.mse_loss(mat_sim, dino_sim)
    return loss

# 2. Smoothness: 相邻edge的k应该相近（除非跨越part边界）
def smoothness_loss(log_k, edge_adjacency, part_same_mask):
    """
    Encourage smooth k within same part.
    """
    diff = (log_k[edge_adjacency[:, 0]] - log_k[edge_adjacency[:, 1]]).abs()
    loss = (diff * part_same_mask).mean()
    return loss
```

### 6.3 完整训练Loss

```python
def compute_total_loss(model, batch, stage="teacher"):
    pred_logk = model(...)

    if stage == "teacher":
        # Stage 1: Teacher supervision
        L_main = F.l1_loss(pred_logk, batch["teacher_logk"])
    else:
        # Stage 2: Physics simulation
        L_main = physics_simulation_loss(pred_logk, batch)

    # Regularization
    L_contrast = contrastive_loss(z_material, batch["part_features"])
    L_smooth = smoothness_loss(pred_logk, batch["edge_adj"], batch["part_same"])

    # Total
    loss = L_main + 0.1 * L_contrast + 0.01 * L_smooth

    return loss
```

---

## 7. Zero-Shot推理

```python
@torch.no_grad()
def zero_shot_inference(model, new_case_path: str) -> Tensor:
    """
    Zero-shot inference for a new case.

    Only requires:
    1. RGB images (for DINO feature extraction)
    2. 3D structure (nodes, edges)
    3. Camera calibration
    """
    # 1. Extract DINO features
    node_dino = extract_dino_features(new_case_path)  # [N, 1024]

    # 2. Cluster nodes into parts
    part_assignments, part_features = cluster_by_dino(node_dino, n_clusters=8)

    # 3. Build edge topology
    edges, z_geo, edge_mid = build_edge_topology(new_case_path)

    # 4. Get part indices for edges
    edge_part_indices = get_edge_parts(edges, part_assignments)

    # 5. Predict physics parameters
    log_k = model(
        part_features,      # [K, 1024]
        edge_part_indices,  # [E, 2]
        z_geo,              # [E, 10]
        edge_mid,           # [E, 3]
    )

    return log_k.exp()  # Return stiffness k
```

---

## 8. 与现有代码的关系

### 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `material_param_dataset.py` | 添加part-level数据加载 |
| `train_paramnet_konly.py` | 替换为新的ZeroShotMaterialPhysics模型 |
| `export_paramnet_teacher.py` | 更新导出逻辑 |

### 新增文件

| 文件 | 内容 |
|------|------|
| `models/zero_shot_physics.py` | ZeroShotMaterialPhysics模型定义 |
| `train_zeroshot.py` | 新的训练脚本 |

### 复用的文件

| 文件 | 复用内容 |
|------|----------|
| `extract_dino_semantic_features.py` | DINO特征提取 |
| `extract_material_parts.py` | Clustering逻辑 |

---

## 9. 关键超参数

```python
# Model architecture
DINO_DIM = 1024           # DINOv2 ViT-L/14 feature dimension
MATERIAL_DIM = 128        # Material embedding dimension (was 16!)
PROJ_HIDDEN = [512, 256]  # Projection network hidden dims
DECODER_HIDDEN = 256      # Physics decoder hidden dim

# Clustering
NUM_PARTS = 8             # Default number of parts per case
PCA_DIM = 32              # PCA before clustering

# Training
LR_STAGE1 = 1e-3          # Learning rate for teacher supervision
LR_STAGE2 = 1e-4          # Learning rate for physics refinement
EPOCHS_STAGE1 = 100       # Teacher supervision epochs
EPOCHS_STAGE2 = 50        # Physics refinement epochs
BATCH_SIZE = 4            # Cases per batch

# Regularization
LAMBDA_CONTRAST = 0.1     # Contrastive loss weight
LAMBDA_SMOOTH = 0.01      # Smoothness loss weight
```

---

## 10. 预期效果

### 优势
1. **Zero-shot泛化**: 新case只需DINO特征，无需重新训练
2. **样本高效**: 只学习投影网络和decoder，参数量小
3. **可解释性**: Part分割和material embedding可视化
4. **利用DINO先验**: DINO的语义信息被有效利用

### 潜在问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| Part数量不一致 | 使用soft assignment或统一的part数量 |
| 跨case的part对齐 | Contrastive loss鼓励相似DINO→相似embedding |
| Teacher noise | Stage 2用物理loss微调 |

---

## 11. 实现的架构: Triplane + Small MLP

为了进一步减少参数量，采用了triplane空间编码：

### 11.1 Triplane Encoder

```
┌─────────────────────────────────────────────────────────────┐
│                    Triplane Spatial Encoding                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   3D Point (x, y, z)                                        │
│         │                                                   │
│         ├──────────────────┬──────────────────┐            │
│         │                  │                  │            │
│         ▼                  ▼                  ▼            │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐        │
│   │ XY Plane │      │ XZ Plane │      │ YZ Plane │        │
│   │ [32x32]  │      │ [32x32]  │      │ [32x32]  │        │
│   │  32-dim  │      │  32-dim  │      │  32-dim  │        │
│   └────┬─────┘      └────┬─────┘      └────┬─────┘        │
│        │                 │                 │               │
│        │    bilinear     │    bilinear     │   bilinear   │
│        │    sample       │    sample       │   sample     │
│        │                 │                 │               │
│        └────────────────┬┴─────────────────┘               │
│                         │                                   │
│                         ▼                                   │
│                  Concatenate                                │
│                  [96-dim]                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Parameters: 3 × 32 × 32 × 32 = 98,304 (vs millions for 3D voxel)
```

### 11.2 完整模型架构

```
┌─────────────────────────────────────────────────────────────┐
│              ZeroShotMaterialPhysics Model                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Inputs:                                                     │
│    edge_mid  [E, 3]    - Edge midpoint coordinates          │
│    z_sem     [E, 1024] - DINO semantic features             │
│    z_geo     [E, 10]   - Geometric features                 │
│                                                              │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────────────┐        │
│  │  TriplaneEncoder │    │  MaterialProjectionNet  │        │
│  │  edge_mid → 96   │    │  z_sem (1024) → 64      │        │
│  │                  │    │                         │        │
│  │  Params: ~98K    │    │  Params: ~280K          │        │
│  └────────┬─────────┘    └───────────┬─────────────┘        │
│           │                          │                       │
│           └───────────┬──────────────┘                       │
│                       │                                      │
│                       ▼                                      │
│              ┌────────────────┐                             │
│              │ PhysicsDecoder │                             │
│              │ [96+64+10]=170 │                             │
│              │     → 128      │                             │
│              │     → 128      │                             │
│              │     → 1        │                             │
│              │                │                             │
│              │ Params: ~40K   │                             │
│              └───────┬────────┘                             │
│                      │                                       │
│                      ▼                                       │
│              log_k [E, 1]                                   │
│                                                              │
│  Total Parameters: ~420K (very compact!)                    │
└─────────────────────────────────────────────────────────────┘
```

### 11.3 参数对比

| 组件 | 旧架构 | 新架构 (Triplane) |
|------|--------|-------------------|
| Spatial encoding | 无 | Triplane ~98K |
| Material embedding | 5×16 = 80 | Projection 1024→64 ~280K |
| Decoder | [1053→256→256→128→1] ~400K | [170→128→128→1] ~40K |
| **Total** | **~400K** | **~420K** |

### 11.4 训练命令

```bash
# 训练 (physics simulation loss, no teacher)
python semantic/train_zeroshot.py \
    --base_path data/different_types \
    --epochs 100 \
    --lr 1e-3 \
    --triplane_resolution 32 \
    --material_dim 64 \
    --exp_name triplane_v1

# 从checkpoint恢复
python semantic/train_zeroshot.py \
    --resume checkpoints/zeroshot/triplane_v1/latest.pth
```

### 11.5 文件结构

```
semantic/
├── models.py              # 模型定义
│   ├── TriplaneEncoder
│   ├── MaterialProjectionNet
│   ├── PhysicsDecoder
│   └── ZeroShotMaterialPhysics
├── train_zeroshot.py      # 训练脚本 (physics loss)
├── train_design.md        # 设计文档
└── ...
```

---

## 12. 实验计划

1. **Baseline**: 当前的MaterialEmbedding + ParamNet
2. **Triplane v1**: Triplane(32) + Material(64) + Decoder(128)
3. **Triplane v2**: Triplane(64) + Material(128) + Decoder(256)
4. **Ablation**: 去掉triplane，只用DINO projection

**评估指标**:
- Physics simulation loss (tracking, geometry, render)
- Zero-shot泛化 (hold-out cases)
- Visual quality of predicted deformation
