'''
file: dataset.py
author: Ale
date: 2026/04/17 
description: 
    - This file contains the dataset class for loading the CIFAR-10 dataset and applying transformations to it. 
    - The class is designed to be used with PyTorch's DataLoader for efficient data loading and batching.
    - The dataset class also includes functionality for visualizing sample images from the dataset.
'''

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
    """
    ---
    Note
    ---
    - There are 13 features taking integer values (mostly count features) and 26 categorical features. 
    - The values of the categorical features have been hashed onto 32 bits for anonymization purposes.
    - The semantic of these features is undisclosed. Some features may have missing values.
    - The rows are chronologically ordered.
    - Format: The columns are tab separeted with the following schema:
        <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
    - When a value is missing, the field is just empty.
    
    ---
    Args
    ---
    - data_path: the path of the data file, e.g. './data/Criteo_small/train.txt'
    
    ---
    Returns
    ---
    - data: a pandas DataFrame containing the processed data, with missing values filled and features transformed.
    - sparse_vocab_size: a dictionary mapping each sparse feature to its vocabulary size (number of unique values).
    - dense_features: a list of the names of the dense features (I1 to I13).
    - sparse_features: a list of the names of the sparse features (C1 to C26).
    
    """
    # --- Step 1: Define feature column names ---
    dense_features = ['I' + str(i) for i in range(1, 14)] # 13 integer/count dense features: I1 to I13
    sparse_features = ['C' + str(i) for i in range(1, 27)] # 26 hashed categorical sparse features: C1 to C26
    col_names = ['label'] + dense_features + sparse_features

    # --- Step 2: Read the raw tab-separated file ---
    # Missing values (empty fields) are automatically parsed as NaN by pandas
    data = pd.read_csv(data_path, sep='\t', header=None, names=col_names)

    # --- Step 3: Fill missing values ---
    # Dense: fill with column median — robust to heavy-tailed count distributions
    #        and avoids distorting the mean with extreme outliers
    data[dense_features] = data[dense_features].fillna(data[dense_features].median())
    
    # Sparse: fill with the string "unknown" so missing values become
    #         a distinct, encodable category rather than silently dropping rows
    data[sparse_features] = data[sparse_features].fillna('unknown')

    # --- Step 4: Normalize dense features to [0, 1] ---
    # MinMaxScaler preserves relative magnitudes and produces bounded inputs
    # suitable for feeding directly into the MLP component of DeepFM
    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])

    # --- Step 5: Encode sparse features as contiguous integer indices ---
    # LabelEncoder maps each unique hash string (and "unknown") to an integer
    # in [0, num_classes - 1], which is the format required by nn.Embedding
    for feat in sparse_features:
        le = LabelEncoder()
        data[feat] = pd.Series(le.fit_transform(data[feat]), index=data.index)


    # --- Step 6: Build vocabulary size dict for each sparse feature ---
    # Computed after encoding; the value equals the number of rows needed in
    # the corresponding embedding table
    sparse_vocab_size = {feat: data[feat].nunique() for feat in sparse_features}

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