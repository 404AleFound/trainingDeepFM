'''
file: model.py
author: Ale
date: 2026/04/17
description:
    - This file contains the implementation of the DeepFM model for click-through rate prediction.
'''

import torch
from basic.layers import LR, FM, MLP, EmbeddingLayer
# from torch_rechub.models.ranking import DeepFM as DeepFM_


"""
For DeepFM model in torch_rechub, the input of the model is a dictionary with the following keys:
- "feat_index": a tensor of shape [batch_size, num_fields] containing the indices of the features for each sample in the batch. The indices should be in the range [0, num_features-1], where num_features is the total number of features in the dataset.
- "feat_value": a tensor of shape [batch_size, num_fields] containing the values of the features for each sample in the batch. The values should be normalized to the range [0, 1] for numerical features and should be 1 for categorical features (since they are represented as one-hot vectors).
- "label": a tensor of shape [batch_size] containing the labels for each sample in the batch. The labels should be binary (0 or 1) for click-through rate prediction tasks.

"""

class DeepFM(torch.nn.Module):
    """
    Deep Factorization Machine Model

    Args:
        deep_features (list): the list of `Feature Class`, training by the deep part module.
        fm_features (list): the list of `Feature Class`, training by the fm part module.
        mlp_params (dict): the params of the last MLP module, keys include:
            `{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, deep_features, fm_features, mlp_params):
        super(DeepFM, self).__init__()
        self.deep_features = deep_features
        self.fm_features = fm_features
        
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])
        self.fm_dims = sum([fea.embed_dim for fea in fm_features])
        
        self.linear = LR(self.fm_dims)  # 1-odrder interaction
        self.fm = FM(reduce_sum=True)  # 2-odrder interaction
        self.embedding = EmbeddingLayer(deep_features + fm_features)
        self.mlp = MLP(self.deep_dims, **mlp_params)

    def forward(self, x):
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  # [batch_size, deep_dims]
        # [batch_size, num_fields, embed_dim]
        input_fm = self.embedding(x, self.fm_features, squeeze_dim=False)

        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y_deep = self.mlp(input_deep)  # [batch_size, 1]
        y = y_linear + y_fm + y_deep
        return torch.sigmoid(y.squeeze(1))


