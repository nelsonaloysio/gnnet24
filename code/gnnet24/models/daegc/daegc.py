import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.typing import Adj, OptTensor
# from sklearn.cluster import KMeans

from .layer import GATLayer


class DAEGC(nn.Module):
    """
    Deep Attentional Embedded Graph Clustering (DAEGC) model with
    a 2-layer Graph Attention Network (GATv1) for pretraining.

    Based on initial implementation from:
        https://github.com/Tiger101010/DAEGC

    Reference papers:
        - [Graph Attention Networks (2018)](https://arxiv.org/abs/1710.10903)
        - [Attributed Graph Clustering: A Deep Attentional Embedding Approach
          (2019)](https://arxiv.org/abs/1906.06532)

    :param in_features: Number of input features.
    :param hidden_size: Number of hidden features.
    :param out_features: Number of output features.
    :param alpha: Alpha for the leaky_relu (default: 0.2).
    """
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        n_clusters: int,
        alpha: float = 0.2,
    ):
        super().__init__()

        self.conv1 = GATLayer(in_features, hidden_size, alpha=alpha)
        self.conv2 = GATLayer(hidden_size, out_features, alpha=alpha)
        self.centroids = Parameter(Tensor(n_clusters, out_features))
        nn.init.xavier_normal_(self.centroids.data)

    def forward(self, x: Tensor, adj: Adj, m: OptTensor = None) -> Tensor:
        """
        Defines forward pass for the graph neural network.

        :param x: Input features.
        :param adj: Adjacency matrix.
        :param m: Precomputed transition matrix (optional).
        """
        h = self.conv1(x, adj, m)
        h = self.conv2(h, adj, m)
        z = F.normalize(h, p=2, dim=1)
        adj_pred = self.dot_product_decode(z)
        return adj_pred, z

    def get_q(self, z: Tensor, how: str = "euclidean", v: int = 1) -> Tensor:
        """
        Compute soft cluster assignments based on the distance between node
        embeddings and cluster centroids.

        :param z: Node embeddings.
        :param how: Distance to use: ('euclidean', 'manhattan', 'cosine').
            Default is 'euclidean'.
        :param v: Hyperparameter for distance calculation.
        """
        if how == "euclidean":
            z = z.unsqueeze(1) - self.centroids
            z = torch.pow(z, 2)
            z = torch.sum(z, 2)

        elif how == "manhattan":
            z = z.unsqueeze(1) - self.centroids
            z = torch.pow(z, 1)
            z = torch.abs(z)
            z = torch.sum(z, 2)

        elif how == "cosine":
            z = torch.nn.CosineSimilarity(dim=2)(z.unsqueeze(1), self.centroids)

        q = 1.0 / (1.0 + z / v)
        q = q.pow((v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def dot_product_decode(z: Tensor) -> Tensor:
        """
        Dot product decoder for link prediction.

        :param z: Node embeddings.
        """
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_pred

    # def save_state_dict(self, path: str) -> None:
    #     """
    #     Save weights to file.
    #
    #     :param path: Path to the file.
    #     """
    #     torch.save(self.state_dict(), path)

    # def load_state_dict(self, path: str, keys: Optional[list] = None, map_location: str = "cpu") -> None:
    #     """
    #     Load weights from file.
    #
    #     :param path: Path to the file.
    #     :param parameters: Parameters to load from the state dict (optional).
    #     :param map_location: Device to map the state dict.
    #     """
    #     checkpoint = torch.load(path, map_location=map_location)
    #     if parameters is None:
    #         parameters = self.state_dict().keys()
    #     for parameter in parameters:
    #         self._parameters[parameter] = checkpoint[parameter]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} " \
               f"({self.in_features} -> {self.out_features})"
