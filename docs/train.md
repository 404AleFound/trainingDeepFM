# 训练流程说明（train.py）

## 整体架构

```
CriteoDataset → DataLoader (collate_fn) → DeepFM → BCELoss → Adam + ReduceLROnPlateau
```

---

## 关键组件

### `build_collate_fn`
将 `CriteoDataset.__getitem__` 返回的 `(label, dense_x, sparse_x)` 元组，在 batch 组装时转换为 `torch_rechub` 要求的特征名 → Tensor 字典格式。

### `train_one_epoch`
- 使用 `tqdm` 显示实时 batch 进度及当前 loss
- **梯度裁剪**：`clip_grad_norm_(max_norm=1.0)`，防止首轮后梯度爆炸导致训练不稳定

### `evaluate`
- 计算 AUC、LogLoss、Accuracy、Precision、Recall、F1
- AUC 是 CTR 任务的核心离线指标（与决策阈值无关）

---

## 超参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `batch_size` | 1024 | 可根据显存调整为 2048/4096 |
| `num_epochs` | 50 | 配合早停使用 |
| `learning_rate` | 1e-3 | Adam 初始学习率 |
| `warmup_epochs` | 2 | 线性 warmup 至 `learning_rate` |
| `weight_decay` | 1e-5 | L2 正则，防止过拟合 |
| `embed_dim` | 32 | 稀疏特征 Embedding 维度 |
| `mlp_dims` | [256, 128, 64] | Deep 部分隐藏层 |
| `dropout` | 0.3 | 针对小数据集过拟合问题提高至 0.3 |
| `early_stop_patience` | 5 | warmup 结束后无提升的最大轮数 |
| `early_stop_min_delta` | 1e-4 | AUC 提升噪声阈值 |

---

## 数据切分策略

使用**时序切分**（末尾 `val_ratio=10%` 作为验证集），而非随机切分。

**原因**：Criteo 数据集按时间顺序排列。若使用随机切分，训练集中会包含验证集的"未来"数据，导致 epoch 1 验证 AUC 虚高，掩盖真实的过拟合现象。

```python
train_set = Subset(dataset, range(train_size))
val_set   = Subset(dataset, range(train_size, len(dataset)))
```

---

## 早停机制

### 问题背景
训练总在首轮达到最佳 AUC，导致早停过早触发（epoch 6 左右）。

### 根本原因分析

| 原因 | 描述 |
|------|------|
| warmup 未被排除 | epoch 1 以半学习率（0.0005）运行并设定 `best_auc`，warmup 结束后 LR 满血才开始，反而难以超越 epoch 1 的 AUC |
| 随机切分数据泄漏 | 验证集含有训练集的"过去"数据，epoch 1 评估结果虚高 |
| 梯度未裁剪 | 首轮后可能出现梯度爆炸，破坏收敛 |

### 修复方案

```python
# warmup 阶段不计入 patience，LR 尚未到位时的不提升属正常现象
if epoch > warmup_epochs:
    if val_auc >= best_auc - early_stop_min_delta:
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop_patience:
            break
```

---

## 学习率调度

采用 **Linear Warmup + Cosine Annealing** 组合策略。

### 常见衰减方式对比

| 方式 | 特点 | 适用场景 |
|------|------|---------|
| `StepLR` | 每 N 轮乘以固定系数 | 需手动指定衰减节点 |
| `ReduceLROnPlateau` | 指标停止提升时才衰减（旧方案） | 收敛不稳定，但依赖验证指标 |
| `ExponentialLR` | 每轮指数衰减 | 衰减较激进 |
| **`CosineAnnealingLR`** | **余弦曲线平滑衰减至 `eta_min`（当前方案）** | warmup + cosine 是现代 DL 标准组合 |
| `CosineAnnealingWarmRestarts` | 余弦 + 周期重启 | 需要跳出局部最优时 |
| `OneCycleLR` | 先升后降的单周期策略 | 训练轮数少、追求快速收敛 |

### 为什么从 ReduceLROnPlateau 换成 CosineAnnealingLR

旧方案存在一个恶性循环：

```
epoch 1 AUC 最高（warmup 半 LR 下的伪最优）
  → epoch 3 连续 2 轮无提升，ReduceLROnPlateau 触发，LR 折半
  → 模型在更低 LR 下更难超越 epoch 1 的 AUC
  → 早停在 epoch 6 左右触发
```

余弦衰减的优势：
1. **确定性**：LR 按固定曲线衰减，与验证指标无关，不会被 epoch 1 的伪最优触发提前衰减
2. **天然配套 warmup**：Warmup + Cosine 是 BERT、ViT 等现代模型的标准调度策略
3. **衰减节奏平滑**：给模型完整 48 轮（`T_max = num_epochs - warmup_epochs`）的学习机会

### 调度曲线

```
LR
1e-3 ──┐ warmup (epoch 1-2)
       └──╮
           ╰──────╮  cosine decay (epoch 3-50)
                   ╰────────────────── 1e-6 (eta_min)
       1   2   3  ...               50  epoch
```

### 实现

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs - warmup_epochs,  # 覆盖 warmup 后的全部轮数
    eta_min=1e-6,                       # LR 下限，避免完全归零
)

# 训练循环中，warmup 结束后每轮调用一次（无需传入指标）
if epoch > warmup_epochs:
    scheduler.step()
```

---

## 输出与日志

- 训练日志同时写入控制台和 `./logger/<timestamp>.log`
- 最优模型保存至 `./checkpoints/deepfm_best.pth`
- 启动时打印模型可训练参数量，用于判断模型是否相对数据集过大
