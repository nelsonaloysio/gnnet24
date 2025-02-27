import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor

from ..utils.abc import ABCLayer, ABCModel


class GAT(ABCModel):
    """
    Implementation of a 2-layer Graph Attention Network (GAT).

    Reference paper:
        - [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (2018)

    :param in_features: Number of input features.
    :param hidden_size: Number of hidden units.
    :param embedding_size: Number of embedding units.
    :param alpha: Alpha for the leaky_relu (default: 0.2).
    """
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()

        self.conv1 = GATLayer(in_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, out_features, alpha)

    def forward(self, x: Tensor, adj: Adj, M: OptTensor = None) -> list:
        """
        Defines forward pass for the graph neural network.

        :param x: Input features.
        :param adj: Adjacency matrix.
        :param M: Precomputed transition matrix (optional).
        """
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z: Tensor) -> Tensor:
        """
        Dot product decoder for link prediction.

        :param Z: Embeddings.
        """
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class GATLayer(ABCLayer):
    """
    Implementation of a single Graph Attention Network layer.

    Reference paper:
        - [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

    :param in_features: Number of input features.
    :param out_features: Number of output features.
    :param alpha: Alpha for the leaky_relu (default: 0.2).
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2) -> None:
        super().__init__(in_features, out_features)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: Tensor, adj: Adj, M: OptTensor = None, concat: bool = True):
        """
        Defines forward pass for the graph neural network.

        :param x: Input features.
        :param adj: Adjacency matrix.
        :param M: Precomputed transition matrix (default: None).
        :param concat: If True, concatenate embeddings (default: True).

        Start by performing a matrix multiplication on input `x` and weights,
        to obtain the embeddings `h` to consider when calculating attention.

        The `attention` is then computed the following way:
            - (1) Obtain self-attention: multiply `h` by weights `a_self` (mm)
            - (2) Obtain neighbor attention: multiply `h` by weights `a_neigh` (mm)
            - (3) Obtain "dense" attention: sum (1) with the transpose of (2)
            - (4) Multiply dense attention (3) by precomputed transition matrix `M`
                  [!] this step only applies to graph autoencoders (e.g., DAEGC)
            - (5) Apply Leaky ReLU on (4) for activation (with predefined `alpha`)

        Then, we compute the embeddings with attention `h_prime`:
            - (1) Initialize an adjacency matrix where, if an edge exists (adj > 0),
                  then the obtained dense attention value is considered (0 otherwise)
            - (2) Normalize values with softmax (normalized exponential function),
                  applying the standard exponential function to each element of the
                  input vector and dividing the result by the sum of all exponentials
            - (3) Perform a `matmul` between obtained attention (2) and `h`

        Once obtained, `h_prime` is computed by multiplying (matmul) `h` and `attention`.
            - https://pytorch.org/docs/stable/generated/torch.matmul.html

        Lastly, if `concat` is True, Exponential Linear Unit (eLU) is used for activation.
        """
        h = torch.mm(x, self.W)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)

        if M is not None:
            attn_dense = torch.mul(attn_dense, M)

        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)

        return h_prime
