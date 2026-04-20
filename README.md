# DeepFM CTR 训练项目

基于 PyTorch 的 DeepFM 实现，用于 Criteo 点击率预估（CTR）任务。

项目特点：
- FM + DNN 联合建模，端到端训练。
- 支持稠密特征多策略处理（统计标准化、MinMax、分桶后转稀疏）。
- 支持稀疏特征哈希编码或词表编码。
- 训练过程输出完整日志，支持自动解析并绘图。
- 保存最优 AUC 的模型权重到 checkpoints。

建议阅读学习笔记：[node.md](./docs/note.md)

## 1. 目录说明

```text
trainingDeepFM/
├── checkpoints/
│   └── deepfm_best.pth
├── data/
│   ├── Criteo/
│   ├── Criteo_sample/
│   └── Criteo_small/
├── docs/
│   ├── note.md
│   └── assets/
├── logger/
├── scripts/
│   ├── basic/
│   │   ├── activation.py
│   │   ├── features.py
│   │   └── layers.py
│   ├── dataset.py
│   ├── model.py
│   ├── plot.py
│   ├── processor.py
│   ├── train.py
│   └── utils.py
├── requrement.txt
└── README.md
```

## 2. 环境安装

建议 Python 版本：3.12。

安装依赖：

```bash
pip install -r requirements.txt
```

说明：
- 文件中已固定 GPU 版 PyTorch（CUDA 12.1）：
  - torch==2.4.1+cu121
  - torchvision==0.19.1+cu121

## 3. 数据准备

### 3.1 数据格式

原始 Criteo 文本按以下字段组织：
- 1 列标签 label
- 13 列连续特征 I1-I13
- 26 列类别特征 C1-C26

### 3.2 预处理流程

预处理在 scripts/processor.py 中完成：
- 自动将 txt 转 csv（若对应 csv 不存在或表头异常）。
- 缺失值填充。
- 连续特征处理（dense_mode）。
- 类别特征处理（sparse_mode）。

可选策略：
- dense_mode:
  - stats
  - minmax
  - bucketize_as_sparse
- sparse_mode:
  - hash
  - lookup

## 4. 训练

在仓库根目录执行：

```bash
python scripts/train.py
```

训练脚本默认设置（可在 scripts/train.py 中修改）：
- batch_size = 1024
- num_epochs = 25
- learning_rate = 5e-4
- warmup_epochs = 5
- scheduler_type = cosine
- early stop 指标：Val AUC
- device：自动使用 cuda（若可用）否则 cpu

训练期间会记录以下指标：
- AUC
- LogLoss
- ACC
- Precision
- Recall
- F1

输出结果：
- 最优模型：checkpoints/deepfm_best.pth
- 训练日志：logger/*.log

## 5. 日志可视化

训练完成后可执行：

```bash
python scripts/plot.py
```

该脚本会自动读取 logger 目录下最新日志并生成图像：
- Step 级别训练损失曲线
- Epoch 级别 Train Loss 与 Val LogLoss
- Epoch 级别 AUC/ACC/F1/Precision/Recall

图像输出目录：logger/plots/

## 6. 核心代码结构

- scripts/model.py：DeepFM 主模型（FM + MLP）。
- scripts/basic/layers.py：EmbeddingLayer、LR、FM、MLP 等基础层。
- scripts/processor.py：数据读取、缺失值处理与特征编码。
- scripts/dataset.py：Dataset 封装。
- scripts/train.py：训练与验证主流程、日志、学习率策略、checkpoint 保存。
- scripts/plot.py：训练日志解析与可视化。

## 7. 常见问题

1. 为什么能读取 txt 训练文件？
processor 会自动将 txt 转换为同名 csv，再进行后续处理。

2. 为什么 GPU 没有被使用？
请确认：
- nvidia-smi 可正常输出。
- torch.cuda.is_available() 返回 True。
- 已安装与本机驱动兼容的 CUDA 版本 PyTorch。

3. 日志里 Precision/Recall/F1 很低是否异常？
CTR 场景通常类别不平衡，需要结合 AUC 与 LogLoss 综合评估，并可调阈值或重采样进一步优化。

## 8. 参考

- 论文：DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
- 学习笔记：docs/note.md