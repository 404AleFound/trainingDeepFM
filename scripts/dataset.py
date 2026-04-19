'''
file: dataset.py
author: Ale
date: 2026/04/17 
description: 
    - This file contains the dataset class for loading the CIFAR-10 dataset and applying transformations to it. 
    - The class is designed to be used with PyTorch's DataLoader for efficient data loading and batching.
    - The dataset class also includes functionality for visualizing sample images from the dataset.
'''
import os
import tqdm
import torch.utils as utils
import doctest
import torch
import torchvision
import torchvision.transforms.v2 as v2
from matplotlib import pyplot as plt  
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# the dir of data - Criteo dataset
# Criteo/
# ├── readme.txt
# ├── text.txt
# └── train.txt
# Criteo_small/
# ├── val.txt
# ├── text.txt
# └── train.txt

def preReadData(data_path):
    '''
    ---
    Note
    ---
    This function reads the data from the given path and returns a list of lines.
    
    ---
    Args    
    ---
    - data_path: the path of the data file, e.g. './data/Criteo_small/train.txt'
    
    '''
    data = []
    with open(data_path, 'r') as f:
        for i in range(5):
            data.append(f.readline())    
    print("The number of samples in the dataset: ", len(data))
    for i in range(5):
        print("Sample {}: {}".format(i, data[i]))


def Process_Data(data_path):
    # --- Step 1: Define feature column names ---
    dense_features = ['I' + str(i) for i in range(1, 14)] # 13 integer/count dense features: I1 to I13
    sparse_features = ['C' + str(i) for i in range(1, 27)] # 26 hashed categorical sparse features: C1 to C26
    col_names = ['label'] + dense_features + sparse_features

    # 定义对应的 Numpy 压缩文件路径
    npz_path = data_path.replace('.txt', '.npz').replace('.csv', '.npz')
    
    if os.path.exists(npz_path):
        print(f"Loading processed data directly from Numpy file: {npz_path}...")
        # 如果存在已经处理好的 npz，使用 np.load 加载
        loaded = np.load(npz_path, allow_pickle=True)
        
        # 将各列重新组装回 DataFrame（这样可以完美保持原本的 int 和 float 数据类型）
        data_dict = {col: loaded[col] for col in col_names}
        data = pd.DataFrame(data_dict)
        
        # 恢复 vocab 字典
        sparse_vocab_size = loaded['vocab'].item()
    else:
        print(f"Raw data processing started from {data_path} (This may take a while)...")
        
        # --- 核心优化 1：使用 chunksize 分块读取以降低峰值内存 ---
        print("Progress: [1~2/4] Reading raw generic TSV file in chunks & Fill NaNs...")
        chunk_list = []
        chunk_size = 500000  # 每次读取 50 万行
        
        # pd.read_csv 的 chunksize 参数会返回一个按块读取的生成器，防止巨大的内存一次性分配
        df_iter = pd.read_csv(data_path, sep='\t', header=None, names=col_names, chunksize=chunk_size)
        
        for chunk in tqdm.tqdm(df_iter, desc="Reading & Processing Chunks"):
            # 在每个 Chunk 内部进行填补，避免读取全部数据后产生巨型 boolean mask 造成极具毁灭性的内存溢出
            # Dense: 为了统一且高效地分配内存，使用 0 填充
            chunk[dense_features] = chunk[dense_features].fillna(0.0)
            # Sparse: 填充 unknown
            chunk[sparse_features] = chunk[sparse_features].fillna('unknown')
            
            # --- 核心优化 2：精确下转数据类型以继续缩小内存基数 ---
            chunk[dense_features] = chunk[dense_features].astype(np.float32)
            chunk['label'] = chunk['label'].astype(np.float32)
            
            chunk_list.append(chunk)

        print("Concatenating chunks into a single DataFrame...")
        data = pd.concat(chunk_list, ignore_index=True)
        del chunk_list # 手动释放列表引用的内存
        
        # --- Step 4: Normalize dense features to [0, 1] ---
        print("Progress: [3/4] Normalizing dense features vs MinMaxScaler...")
        scaler = MinMaxScaler()
        data[dense_features] = scaler.fit_transform(data[dense_features])
        data[dense_features] = data[dense_features].astype(np.float32)

        # --- Step 5: Encode sparse features ---
        print("Progress: [4/4] Encoding sparse features with pd.factorize (Memory Optimized)...")
        sparse_vocab_size = {}
        for feat in tqdm.tqdm(sparse_features, desc="Encoding Sparse Features"):
            # 核心优化 3: 弃用极其耗费空间储存的 sklearn.LabelEncoder
            # 采用 Pandas 底层的 factorize 算法不仅速度快 3~5 倍，而且内存占用极低
            data[feat], uniques = pd.factorize(data[feat])
            data[feat] = data[feat].astype(np.int32)
            sparse_vocab_size[feat] = len(uniques)

        # 存为 npz 压缩文件
        print(f"Saving the processed DataFrame to {npz_path} for faster future loading...")
        # 将每一列单独存入，外加 vocab 的 dict
        npz_dict = {col: data[col].values for col in col_names}
        npz_dict['vocab'] = np.array(sparse_vocab_size)
        np.savez_compressed(npz_path, **npz_dict)

    return data, sparse_vocab_size, dense_features, sparse_features



class CriteoDataset(utils.data.Dataset):
    """
    ---
    Note
    ---
    PyTorch Dataset wrapper for the Criteo CTR prediction dataset.

    Calls Process_Data internally to handle missing-value imputation,
    dense feature normalization, and sparse feature label-encoding.
    Each sample is returned as three tensors ready for DeepFM input:
        - label    : float32 scalar (0 or 1)
        - dense_x  : float32 vector of shape [13]  (normalized dense features I1-I13)
        - sparse_x : long    vector of shape [26]  (encoded sparse feature indices C1-C26)
    
    ---
    Args
    ---
        data_path (str): path to the raw Criteo TSV file,
                         e.g. './data/Criteo_small/train.txt'
    ---
    Example
    ---
        >>> dataset = CriteoDataset('./data/Criteo_small/train.txt')
        >>> label, dense_x, sparse_x = dataset[0]
        >>> dense_x.shape, sparse_x.shape
        (torch.Size([13]), torch.Size([26]))
    """

    def __init__(self, data_path: str) -> None:
        # Process the raw file once at construction time so __getitem__
        # only needs a fast DataFrame row-lookup during training
        self.data, self.parse_vocab_size, self.dense_features, self.sparse_features = Process_Data(data_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        # Binary click label: 0 or 1
        label = torch.tensor(float(row['label']), dtype=torch.float32)

        # Normalized dense features — float32 for direct MLP input
        dense_x = torch.tensor(
            row[self.dense_features].values.astype(float),
            dtype=torch.float32
        )

        # Encoded sparse features indices — long (int64) for nn.Embedding lookup
        sparse_x = torch.tensor(
            row[self.sparse_features].values.astype(int),
            dtype=torch.long
        )

        return label, dense_x, sparse_x
    
if __name__ == "__main__":
    # test the dataset
    dataset = CriteoDataset('./data/Criteo_small/train.txt')
    label, dense_x, sparse_x = dataset[0]
    print("Label: ", label)
    print("Dense features: ", dense_x)
    print("Sparse features: ", sparse_x)