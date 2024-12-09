import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor


class GATLayer(nn.Module):
    """
    Single Graph Attention Network layer, similar to the one
    introduced in [GATv1](https://arxiv.org/abs/1710.10903).

    :param in_features: Number of input features.
    :param out_features: Number of output features.
    :param alpha: Alpha for Leaky ReLU (default: 0.2).
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.w = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: Tensor, adj: Adj, m: OptTensor = None, elu: bool = True):
        """
        Defines forward pass for the graph neural network.

        :param x: Input features.
        :param adj: Adjacency matrix.
        :param m: Precomputed transition matrix. Optional.
        :param elu: Concatenate and apply eLU for activation. Default is True.

        Start by performing a matrix multiplication on input (x) and weights,
        to obtain the embeddings (h) to consider when calculating attention.

        The attention is calculated in the following steps:
            - (1) Obtain self-attention: (h * a_self).
            - (2) Obtain neighbor attention: (h * a_mult).
            - (3) Obtain dense attention: sum (1) and (2) transposed.
            - (4) Optionally multiply dense attention (3) by transition matrix.
            - (5) Apply Leaky ReLU activation to the dense attention.

        The final embeddings are computed in the following steps:
            - (1) Initialize an adjacency matrix where, if an edge exists,
                  the obtained dense attention value is used, and 0 otherwise.
            - (2) Normalize with softmax (normalized exponential function),
                  applying the standard exponential function to each element of
                  the matrix and dividing by the sum of the elements in the row.
            - (3) Multiply the normalized attention matrix with the embeddings.
            - (4) If `elu` is True, apply Exponential Linear Unit (eLU).
        """
        h = torch.mm(x, self.w)

        attn_for_self = torch.mm(h, self.a_self)
        attn_for_neighs = torch.mm(h, self.a_neighs)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        if m is not None:
            attn_dense = torch.mul(attn_dense, m)
        attn_dense = self.leakyrelu(attn_dense)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if elu:
            return F.elu(h_prime)

        return h_prime

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} " \
               f"({self.in_features} -> {self.out_features})"
