from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from torch.nn.parameter import Parameter
from torch_geometric.typing import Adj, PairTensor
# from sklearn.preprocessing import normalize


class DAEGC(nn.Module):
    """
    Deep Autoencoder for Graph Clustering (DAEGC) model.

    Initial implementation from Tiger101010:
        https://github.com/Tiger101010/DAEGC
    """
    def __init__(
        self,
        n_features: int,
        n_clusters: int,
        distance: str = "euclidean",
        v: int = 1,
    ):
        super().__init__()

        self.n_clusters = n_clusters
        self.distance = distance
        self.v = v

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_features))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, Z: torch.Tensor):
        """ Forward pass. """
        Q = self.get_Q(Z)
        return Q

    @staticmethod
    def get_A(edge_index: Adj, weighted: bool = False) -> torch.Tensor:
        """ Get dense adjacency matrix from edge index. """
        A = pyg.utils.to_dense_adj(edge_index)[0]
        if not weighted:
            A[A != 0] = 1
        return A

    staticmethod
    def get_B(adj: Adj) -> torch.Tensor:
        """ Get normalized adjacency matrix with positive diagonals. """
        w = DAEGC.get_w(adj)
        return (adj + torch.eye(adj.shape[0])) * w.reshape(-1, 1)

    @staticmethod
    def get_M(adj: Adj, t: int = 2, norm="l1"):
        """ Get proximity matrix from adjacencies. """
        # adj = normalize(adj, norm=norm, axis=0)
        return torch.from_numpy(
            sum([np.linalg.matrix_power(adj, i) for i in range(1, t + 1)]) / t
        )

    @staticmethod
    def get_P(q: torch.Tensor) -> torch.Tensor:
        """ Get target probability distribution. """
        weight = q**2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p

    def get_Q(self, Z: torch.Tensor) -> torch.Tensor:
        """ Get soft cluster assignments. """
        if self.distance == "euclidean":
            q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)

        elif self.distance == "manhattan":
            q = 1.0 / (1.0 + torch.sum(torch.abs(torch.pow(Z.unsqueeze(1) - self.cluster_layer, 1)), 2) / self.v)

        elif self.distance == "cosine":
            q = 1.0 / (1.0 + torch.nn.CosineSimilarity(dim=2)(Z.unsqueeze(1), self.cluster_layer) / self.v)

        # elif self.distance == "mass":
        #     q = 1.0 / (1.0 + mass_dissimilarity(Z)

        else:
            raise ValueError("Invalid distance, choose from ['euclidean', 'manhattan', 'cosine'].")

        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def get_w(adj: Union[Adj, PairTensor], dtype=None) -> np.ndarray:
        """ Get node weight vector from adjacencies. """
        d = pyg.utils.degree(adj[0]) if (adj.shape[0] == 2 != adj.shape[1]) else adj.sum(axis=0)
        return 1 / (d.to(dtype=dtype).numpy() + 1)
