"""
file: dataset.py
author: Ale
date: 2026/04/19

说明：
    - 仅保留 Dataset 封装
    - 数据处理逻辑迁移到 data.py
"""

import torch
import torch.utils as utils

from processor import (
    CriteoDataProcessor,
    CONTINUOUS_COLUMNS,
    CATEGORICAL_COLUMNS,
    LABEL_COLUMN,
)


class CriteoDataset(utils.data.Dataset):
    """
    Criteo CTR 数据集封装：
    - 调用 processor.py 中的处理流程
    - dense_mode: stats（使用训练集统计量填充）或 zero（使用0填充）
    - sparse_mode: hash（使用哈希编码）或 category（使用类别编码）
    """

    def __init__(self, data_path: str, processor: CriteoDataProcessor) -> None:
        self.processor = processor
        self.data = self.processor.load_and_process(data_path)
        self.dense_features = CONTINUOUS_COLUMNS
        self.sparse_features = CATEGORICAL_COLUMNS

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        label = torch.tensor(float(row[LABEL_COLUMN]), dtype=torch.float32)
        dense_x = torch.tensor(row[self.dense_features].values.astype(float), dtype=torch.float32)
        sparse_x = torch.tensor(row[self.sparse_features].values.astype(int), dtype=torch.long)

        return label, dense_x, sparse_x


if __name__ == "__main__":
    processor = CriteoDataProcessor()
    dataset = CriteoDataset("./data/Criteo_small/train.txt", processor)
    label, dense_x, sparse_x = dataset[0]
    print("Label: ", label)
    print("Dense features: ", dense_x)
    print("Sparse features: ", sparse_x)
