import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import activation_layer
from .features import DenseFeature, SequenceFeature, SparseFeature

class ConcatPooling(nn.Module):
    """Keep original sequence embedding shape.

    Shape
    -----
    Input: ``(B, L, D)``  
    Output: ``(B, L, D)``
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class AveragePooling(nn.Module):
    """Mean pooling over sequence embeddings.

    Shape
    -----
    Input
        x : ``(B, L, D)``
        mask : ``(B, 1, L)``
    Output
        ``(B, D)``
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class SumPooling(nn.Module):
    """Sum pooling over sequence embeddings.

    Shape
    -----
    Input
        x : ``(B, L, D)``
        mask : ``(B, 1, L)``
    Output
        ``(B, D)``
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)


class EmbeddingLayer(nn.Module):
    """General embedding layer.

    Stores per-feature embedding tables in ``embed_dict``.

    Parameters
    ----------
    features : list
        Feature objects to create embedding tables for.

    Shape
    -----
    Input
        x : dict
            ``{feature_name: feature_value}``; sequence values shape ``(B, L)``,
            sparse/dense values shape ``(B,)``.
        features : list
            Feature list for lookup.
        squeeze_dim : bool, default False
            Whether to flatten embeddings.
    Output
        - Dense only: ``(B, num_dense)``.
        - Sparse: ``(B, num_features, embed_dim)`` or flattened.
        - Sequence: same as sparse or ``(B, num_seq, L, embed_dim)`` when ``pooling="concat"``.
        - Mixed: flattened sparse plus dense when ``squeeze_dim=True``.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0
        self.input_mask = InputMask()

        for fea in features:
            if fea.name in self.embed_dict:  # exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with is None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with is None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with is None:
                    sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
                else:
                    sparse_emb.append(self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." % (["sum", "mean"], fea.pooling))
                fea_mask = self.input_mask(x, fea)
                if fea.shared_with is None:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.name](x[fea.name].long()), fea_mask).unsqueeze(1))
                else:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long()), fea_mask).unsqueeze(1))  # shared specific sparse feature embedding
            else:
                dense_values.append(x[fea.name].float() if x[fea.name].float().dim() > 1 else x[fea.name].float().unsqueeze(1))  # .unsqueeze(1).unsqueeze(1)

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            # TODO: support concat dynamic embed_dim in dim 2
            # [batch_size, num_features, embed_dim]
            sparse_emb = torch.cat(sparse_emb, dim=1)

        if squeeze_dim:  # Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  # only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                # squeeze dim to : [batch_size, num_features*embed_dim]
                return sparse_emb.flatten(start_dim=1)
            elif dense_exists and sparse_exists:
                # concat dense value with sparse embedding
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values), dim=1)
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                return sparse_emb  # [batch_size, num_features, embed_dim]
            else:
                raise ValueError("If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" % ("SparseFeatures", features))


class InputMask(nn.Module):
    """Return input masks from features.

    Shape
    -----
    Input
        x : dict
            ``{feature_name: feature_value}``; sequence ``(B, L)``, sparse/dense ``(B,)``.
        features : list or SparseFeature or SequenceFeature
            All elements must be sparse or sequence features.
    Output
        - Sparse: ``(B, num_features)``
        - Sequence: ``(B, num_seq, seq_length)``
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        for fea in features:
            if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature):
                if fea.padding_idx is not None:
                    fea_mask = x[fea.name].long() != fea.padding_idx
                else:
                    fea_mask = x[fea.name].long() != -1
                mask.append(fea_mask.unsqueeze(1).float())
            else:
                raise ValueError("Only SparseFeature or SequenceFeature support to get mask.")
        return torch.cat(mask, dim=1)


class LR(nn.Module):
    """Logistic regression module.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    sigmoid : bool, default False
        Apply sigmoid to output when True.

    Shape
    -----
    Input: ``(B, input_dim)``
    Output: ``(B, 1)``
    """

    def __init__(self, input_dim, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)

class MLP(nn.Module):
    """Multi-layer perceptron with BN/activation/dropout per linear layer.

    Parameters
    ----------
    input_dim : int
        Input dimension of the first linear layer.
    output_layer : bool, default True
        If True, append a final Linear(*,1).
    dims : list, default []
        Hidden layer sizes.
    dropout : float, default 0
        Dropout probability.
    activation : str, default 'relu'
        Activation function (sigmoid, relu, prelu, softmax).

    Shape
    -----
    Input: ``(B, input_dim)``  
    Output: ``(B, 1)`` or ``(B, dims[-1])``
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FM(nn.Module):
    """Factorization Machine for 2nd-order interactions.

    Parameters
    ----------
    reduce_sum : bool, default True
        Sum over embed dim (inner product) when True; otherwise keep dim.

    Shape
    -----
    Input: ``(B, num_features, embed_dim)``  
    Output: ``(B, 1)`` or ``(B, embed_dim)``
    """

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


