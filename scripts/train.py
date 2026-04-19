'''
file: train.py
author: Ale
date: 2026/04/17
description:
    - DeepFM 训练主流程（Criteo CTR 数据集）
    - 评估指标：AUC / LogLoss / ACC / Precision / Recall / F1
    - 训练策略：Linear Warmup + CosineAnnealing + Early Stopping
'''

import logging
import os
import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm
from model import DeepFM
from basic.features import DenseFeature, SparseFeature
from utils import seed_everything
from dataset import CriteoDataset
from processor import CATEGORY_FEATURE_STATS, CriteoDataProcessor

seed_everything(42)

# 创建 logger 文件夹并生成带时间戳的日志文件名
os.makedirs('./logger', exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('./logger', f'{timestamp}.log')

# 日志同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

file_only_logger = logging.getLogger('file_only')
file_only_logger.setLevel(logging.INFO)
file_only_handler = logging.FileHandler(log_file, encoding='utf-8')
file_only_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
file_only_logger.addHandler(file_only_handler)
file_only_logger.propagate = False


# ------------------------------
# DataLoader 组装函数
# ------------------------------

def build_collate_fn(dense_features: list, sparse_features: list):
    """
    将 Dataset 返回的 (label, dense_x, sparse_x) 组装为模型输入字典。

    返回:
        可传入 DataLoader 的 collate_fn。
    """
    def _collate(batch):
        labels = torch.stack([item[0] for item in batch])
        dense_x = torch.stack([item[1] for item in batch])
        sparse_x = torch.stack([item[2] for item in batch])

        x: dict[str, torch.Tensor] = {}
        for i, name in enumerate(dense_features):
            x[name] = dense_x[:, i]
        for i, name in enumerate(sparse_features):
            x[name] = sparse_x[:, i]

        return x, labels

    return _collate


# ------------------------------
# 训练 / 评估函数
# ------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: data_utils.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """执行一个 epoch 的训练，并返回平均 BCE loss。"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc='  Train', leave=False, unit='batch')
    for step, (x, labels) in enumerate(pbar):
        x      = {k: v.to(device) for k, v in x.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, labels)
        loss.backward()
        # 梯度裁剪，避免训练不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        
        file_only_logger.info(f"Epoch [{epoch}] Step [{step}/{len(loader)}] Train Loss: {loss.item():.4f}")
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: data_utils.DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float, float]:
    """在评估集上计算 AUC/LogLoss/ACC/Precision/Recall/F1。"""
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []

    for x, labels in tqdm(loader, desc='  Eval ', leave=False, unit='batch'):
        x = {k: v.to(device) for k, v in x.items()}
        preds = model(x).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, bin_preds)
    pre = precision_score(all_labels, bin_preds, zero_division=0)
    rec = recall_score(all_labels, bin_preds, zero_division=0)
    f1 = f1_score(all_labels, bin_preds, zero_division=0)
    
    return float(auc), float(logloss), float(acc), float(pre), float(rec), float(f1)


# ------------------------------
# 主流程
# ------------------------------

def main() -> None:
    # 超参数
    train_path = './data/Criteo_sample/train_1m_time.txt'
    val_path = './data/Criteo_sample/val_100k_time.txt'
    ckpt_dir = './checkpoints'
    batch_size = 1024 * 4
    num_workers = 16
    num_epochs = 100
    learning_rate = 5e-3
    min_lr = 1e-6
    warmup_epochs = 5
    scheduler_type = 'cosine'   # 'cosine' 或 'none'
    dense_mode = 'bucketize_as_sparse'  # 将所有稠密特征分桶并当作类别特征 bucketize_as_sparse
    sparse_mode = 'hash'        # 'hash' 或 'lookup'
    fill_dense = 0.0
    fill_sparse = 'unknown'
    weight_decay = 1e-3
    embed_dim = 32
    mlp_dims = [256, 128, 64]
    dropout = 0.5
    early_stop_patience = 15
    early_stop_min_delta = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = 6666

    os.makedirs(ckpt_dir, exist_ok=True)

    # 记录超参数，便于复现与对比
    hyperparams = {
        'train_path': train_path,
        'val_path': val_path,
        'ckpt_dir': ckpt_dir,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'min_lr': min_lr,
        'warmup_epochs': warmup_epochs,
        'scheduler_type': scheduler_type,
        'weight_decay': weight_decay,
        'embed_dim': embed_dim,
        'mlp_dims': mlp_dims,
        'dropout': dropout,
        'early_stop_patience': early_stop_patience,
        'early_stop_min_delta': early_stop_min_delta,
        'dense_mode': dense_mode,
        'sparse_mode': sparse_mode,
        'fill_dense': fill_dense,
        'fill_sparse': fill_sparse,
    }
    pretty_hparams = json.dumps(
        hyperparams,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    logger.info('Hyperparameters:\n%s', pretty_hparams)

    # 数据集
    logger.info('Loading and processing dataset...')
    processor = CriteoDataProcessor(
        dense_mode=dense_mode,
        sparse_mode=sparse_mode,
        fill_dense=fill_dense,
        fill_sparse=fill_sparse,
    )
    train_set = CriteoDataset(train_path, processor)
    val_set = CriteoDataset(val_path, processor)

    train_size = len(train_set)
    val_size = len(val_set)
    dense_feas = train_set.dense_features
    print(f"Dense features: {dense_feas[0:5]} ... {dense_feas[-5:]} (total {len(dense_feas)})")
    sparse_feas = train_set.sparse_features
    print(f"Sparse features: {sparse_feas[0:5]} ... {sparse_feas[-5:]} (total {len(sparse_feas)})")
    vocab = {name: CATEGORY_FEATURE_STATS[name] for name in sparse_feas}
    collate_fn = build_collate_fn(dense_feas, sparse_feas)
    
    # DataLoader
    train_loader = data_utils.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
    )
    val_loader = data_utils.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
    )

    # 特征定义
    use_bucketized_dense = (dense_mode == 'bucketize_as_sparse')

    if use_bucketized_dense:
        dense_feature_objs = [
            SparseFeature(
                name,
                vocab_size=processor.dense_processors[name].vocab_size,
                embed_dim=embed_dim,
            )
            for name in dense_feas
        ]
    else:
        dense_feature_objs = [DenseFeature(name) for name in dense_feas]

    sparse_feature_objs = [
        SparseFeature(name, vocab_size=vocab[name], embed_dim=embed_dim)
        for name in sparse_feas
    ]

    deep_feature_objs = dense_feature_objs + sparse_feature_objs
    print(f"Deep features: {[fea.name for fea in deep_feature_objs]} (total {len(deep_feature_objs)})")
    # 分桶后的 dense 会作为 sparse 参与 FM；否则 FM 仅使用原始 sparse。
    fm_feature_objs = deep_feature_objs if use_bucketized_dense else sparse_feature_objs
    print(f"FM features: {[fea.name for fea in fm_feature_objs]} (total {len(fm_feature_objs)})")

    # 模型
    model = DeepFM(
        deep_features=deep_feature_objs,
        fm_features=fm_feature_objs,
        mlp_params={'dims':mlp_dims, 'dropout':dropout, 'activation': 'relu',}
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model trainable parameters: {num_params:,}')

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    # 余弦衰减策略：warmup 后平滑衰减到 min_lr
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=min_lr,
        )

    def set_lr(new_lr: float) -> None:
        for group in optimizer.param_groups:
            group['lr'] = new_lr

    if warmup_epochs > 0:
        set_lr(0.0)

    logger.info(
        f'Device : {device} Train  : {train_size:,} samples Val    : {val_size:,} samples'
    )
    logger.info(
        f'LR Strategy: {scheduler_type} | Base LR: {learning_rate:.2e} | Min LR: {min_lr:.2e} | Warmup: {warmup_epochs}'
    )

    # 训练循环
    best_auc = float('-inf')
    no_improve_epochs = 0
    for epoch in range(1, num_epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            # 线性 warmup
            warmup_lr = learning_rate * (epoch / warmup_epochs)
            set_lr(warmup_lr)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_auc, val_logloss, val_acc, val_pre, val_rec, val_f1 = evaluate(model, val_loader, device)
        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        logger.info(
            f'Epoch [{epoch:>2}/{num_epochs}] | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val AUC: {val_auc:.4f} | '
            f'LogLoss: {val_logloss:.4f} | '
            f'ACC: {val_acc:.4f} | '
            f'Pre: {val_pre:.4f} | '
            f'Rec: {val_rec:.4f} | '
            f'F1: {val_f1:.4f} | '
            f'LR: {lr_now:.2e}'
        )

        # 保存最优 checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(ckpt_dir, 'deepfm_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f'=> Best model saved  (AUC={best_auc:.4f}, LogLoss={val_logloss:.4f}, F1={val_f1:.4f})')

        # warmup 结束后再计早停耐心值
        if epoch > warmup_epochs:
            if val_auc >= best_auc - early_stop_min_delta:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                logger.info(f'=> No AUC improvement for {no_improve_epochs} epoch(s)')
                if no_improve_epochs >= early_stop_patience:
                    logger.info(f'=> Early stopping triggered after {early_stop_patience} epochs without improvement')
                    break

    logger.info(f'\nTraining complete.  Best Val AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()