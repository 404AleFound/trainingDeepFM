'''
file: train.py
author: Ale
date: 2026/04/17
description:
    - Full training pipeline for DeepFM on the Criteo CTR dataset.
    - Uses torch_rechub's DeepFM with DenseFeature / SparseFeature abstractions.
    - Evaluates with AUC (the standard offline metric for CTR prediction).
    - Saves the best checkpoint and applies warmup + cosine annealing scheduling.
'''

import logging
import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, 
    precision_score, recall_score, f1_score
)
from tqdm import tqdm
from typing import Tuple, List
from torch_rechub.models.ranking import DeepFM
from torch_rechub.basic.features import DenseFeature, SparseFeature
from utils import seed_everything
from dataset import CriteoDataset

seed_everything(42)  # 确保全局随机种子固定，结果可复现

# 创建 logger 文件夹并生成带时间戳的日志文件名
os.makedirs('./logger', exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('./logger', f'{timestamp}.log')

# Configure a module-level logger with timestamp + level + message layout.
# basicConfig is called once here so any handler added by the caller is not
# overridden; if the root logger already has handlers this call is a no-op.
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


# ------------------------------------------------------------------ #
#  Collate function                                                    #
# ------------------------------------------------------------------ #

def build_collate_fn(dense_features: list, sparse_features: list):
    """
    Build a collate function for DataLoader.

    CriteoDataset.__getitem__ returns (label, dense_x, sparse_x) where
    dense_x/sparse_x are flat tensors indexed by position.  torch_rechub's
    EmbeddingLayer expects a dict keyed by feature name.  This function
    converts between the two formats at batch-collation time.

    Args:
        dense_features  (list[str]): ordered dense column names, e.g. ['I1',...,'I13']
        sparse_features (list[str]): ordered sparse column names, e.g. ['C1',...,'C26']

    Returns:
        callable: collate_fn suitable for torch.utils.data.DataLoader
    """
    def _collate(batch):
        labels   = torch.stack([item[0] for item in batch])   # [B]
        dense_x  = torch.stack([item[1] for item in batch])   # [B, 13]
        sparse_x = torch.stack([item[2] for item in batch])   # [B, 26]

        # Unpack each column into a named entry; the model indexes by name
        x: dict[str, torch.Tensor] = {}
        for i, name in enumerate(dense_features):
            x[name] = dense_x[:, i]    # [B], float32
        for i, name in enumerate(sparse_features):
            x[name] = sparse_x[:, i]   # [B], long

        return x, labels

    return _collate


# ------------------------------------------------------------------ #
#  Train / evaluate helpers                                            #
# ------------------------------------------------------------------ #

def train_one_epoch(
model: nn.Module,
    loader: data_utils.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Run one full pass over the training set.

    Returns:
        float: mean BCE loss across all mini-batches.
    """
    model.train()
    total_loss = 0.0

    # leave=False: bar disappears after the epoch finishes, keeping the log clean
    pbar = tqdm(loader, desc='  Train', leave=False, unit='batch')
    for step, (x, labels) in enumerate(pbar):
        x      = {k: v.to(device) for k, v in x.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(x)                   # [B], already sigmoid-activated
        loss  = criterion(preds, labels)
        loss.backward()
        # Clip gradients to prevent large updates that destabilise training
        # after the first epoch. max_norm=1.0 is a standard safe default.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        
        # 记录当前 step 的 loss，仅输出到文件
        file_only_logger.info(f"Epoch [{epoch}] Step [{step}/{len(loader)}] Train Loss: {loss.item():.4f}")
        
        # Update the postfix so the current batch loss is visible in real time
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: data_utils.DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute AUC, LogLoss, Accuracy, Precision, Recall, and F1-Score.

    Returns:
        tuple: (AUC, LogLoss, Accuracy, Precision, Recall, F1-Score).
    """
    model.eval()
    all_preds:  list[float] = []
    all_labels: list[float] = []

    for x, labels in tqdm(loader, desc='  Eval ', leave=False, unit='batch'):
        x     = {k: v.to(device) for k, v in x.items()}
        preds = model(x).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    
    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, bin_preds)
    pre = precision_score(all_labels, bin_preds, zero_division=0)
    rec = recall_score(all_labels, bin_preds, zero_division=0)
    f1  = f1_score(all_labels, bin_preds, zero_division=0)
    
    return auc, logloss, acc, pre, rec, f1


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    # ---- Hyperparameters ------------------------------------------ #
    train_path    = './data/Criteo_sample/train_sample.txt'  # 请确保这个路径指向正确的训练数据文件
    ckpt_dir      = './checkpoints'
    batch_size    = 1024 * 1       # 8GB VRAM 比较充裕，可以提升到 2048 或 4096 提高 GPU 吞吐
    num_epochs    = 50          # 轮数稍增加
    learning_rate = 1e-3
    warmup_epochs = 2           # linear warmup steps (in epochs)
    weight_decay  = 1e-5        # [新增] 防止过拟合的权重衰减
    embed_dim     = 32          # 典型的 CTR Embedding 维度推荐用 16-64 之间
    mlp_dims      = [256, 128, 64] # 加深了网络
    dropout       = 0.3         # 根据刚才的严重过拟合，提高到 0.3
    val_ratio     = 0.1         # fraction of training data held out for validation
    early_stop_patience = 5     # stop if no AUC improvement for N epochs
    early_stop_min_delta = 1e-4 # minimum AUC gain to reset patience
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Dataset -------------------------------------------------- #
    # IMPORTANT: load the full file once and split afterwards.
    # Loading train/val from separate files would run LabelEncoder
    # independently on each file, so the same hash string could receive
    # a different integer index in val — corrupting the embedding lookup.
    # Splitting a single dataset guarantees both subsets share the same
    # encoder mapping fitted in Process_Data.
    logger.info('Loading and processing dataset...')
    dataset = CriteoDataset(train_path)

    dense_feas  = dataset.dense_features     # ['I1', ..., 'I13']
    sparse_feas = dataset.sparse_features    # ['C1', ..., 'C26']
    vocab       = dataset.parse_vocab_size  # {name: num_unique_values}

    val_size   = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    # Temporal split: last val_ratio% of rows become validation.
    # Criteo rows are chronologically ordered; random split leaks future
    # data into training and inflates epoch-1 validation AUC artificially.
    train_set = data_utils.Subset(dataset, range(train_size))
    val_set   = data_utils.Subset(dataset, range(train_size, len(dataset)))

    collate_fn = build_collate_fn(dense_feas, sparse_feas)

    # num_workers=0: avoids multiprocess pickling issues on Windows;
    # pin_memory speeds up host→GPU transfers when CUDA is available
    train_loader = data_utils.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
    )
    val_loader = data_utils.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
    )

    # ---- Feature definitions -------------------------------------- #
    # DenseFeature  : passes the scalar value directly into the MLP
    # SparseFeature : looks up a learned embed_dim-D embedding vector
    dense_feature_objs  = [DenseFeature(name) for name in dense_feas]
    sparse_feature_objs = [
        SparseFeature(name, vocab_size=vocab[name], embed_dim=embed_dim)
        for name in sparse_feas
    ]

    # ---- Model ---------------------------------------------------- #
    # FM   part : sparse features only — captures 2nd-order pairwise interactions
    # Deep part : dense + sparse — captures higher-order non-linear interactions
    model = DeepFM(
        deep_features=dense_feature_objs + sparse_feature_objs,
        fm_features=sparse_feature_objs,
        mlp_params={
            'dims':       mlp_dims,
            'dropout':    dropout,
            'activation': 'relu',
        },
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model trainable parameters: {num_params:,}')

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Cosine annealing over the post-warmup epochs.
    # Paired with the linear warmup above, this forms the standard
    # "warmup + cosine decay" schedule used in modern deep learning.
    # Advantages over ReduceLROnPlateau:
    #   - Deterministic: LR follows a fixed curve regardless of val metric.
    #   - Avoids the negative interaction where plateau detection fires at
    #     epoch 3 (because epoch-1 AUC was best), halving LR before the
    #     model has had a fair chance to learn at full LR.
    # T_max = remaining epochs after warmup so the cosine reaches eta_min
    # exactly at the last epoch.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6,
    )

    def set_lr(new_lr: float) -> None:
        for group in optimizer.param_groups:
            group['lr'] = new_lr

    if warmup_epochs > 0:
        set_lr(0.0)

    logger.info(
        f'Device : {device} Train  : {train_size:,} samples Val    : {val_size:,} samples'
    )

    # ---- Training loop -------------------------------------------- #
    best_auc = float('-inf')
    no_improve_epochs = 0
    for epoch in range(1, num_epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            # Linear warmup from 0 to learning_rate
            warmup_lr = learning_rate * (epoch / warmup_epochs)
            set_lr(warmup_lr)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_auc, val_logloss, val_acc, val_pre, val_rec, val_f1 = evaluate(model, val_loader, device)
        if epoch > warmup_epochs:
            scheduler.step()   # CosineAnnealingLR needs no metric argument

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

        # Persist checkpoint if this is the best AUC seen so far.
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(ckpt_dir, 'deepfm_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f'=> Best model saved  (AUC={best_auc:.4f}, LogLoss={val_logloss:.4f}, F1={val_f1:.4f})')

        # Count patience only AFTER warmup -- during warmup the LR is still
        # rising, so non-improving epochs are expected and must not burn patience.
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