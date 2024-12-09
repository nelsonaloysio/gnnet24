import torch
import torch.nn.functional as F
from torch_geometric.typing import Adj, PairTensor, Tensor
from torch_geometric.utils import to_dense_adj


def preprocess(data, t: int = 2, weights: bool = True, device: str = "cpu"):
    """
    Preprocess data for DAEGC model.

    :param data: Data to preprocess.
    :param t: Consider t-hop neighbors for proximity matrix.
    :param weights: Whether to consider edge weights.
    :param device: Device to use.
    """
    # Adjacency matrix with self-loops A = A + I.
    data.adj = to_dense_adj(data.edge_index)[0]
    data.adj.fill_diagonal_(1)

    if not weights:
        data.adj[data.adj > 0] = 1

    # Normalized adjacency matrix Â = I + D^{-1}A.
    data.adj_norm = F.normalize(data.adj, p=1, dim=0)
    b = F.normalize(data.adj_norm, p=1, dim=1)

    # Proximity matrix with t-hop neighbors M = (B + B^2 + ... + B^t) / t.
    data.adj_prox = sum([
        torch.matrix_power(b, i) for i in range(1, t + 1)]) / t

    # Move data to device.
    data.x = data.x.to(device, dtype=torch.float32)
    data.y = data.y.cpu().numpy()
    data.adj = data.adj.cpu()
    data.adj_norm = data.adj_norm.to(device, dtype=torch.float32)
    data.adj_prox = data.adj_prox.to(device, dtype=torch.float32)

    if "train_mask" not in data:
        data.train_mask = torch.ones(data.x.shape[0], dtype=bool)
    if "val_mask" not in data:
        data.val_mask = torch.zeros(data.x.shape[0], dtype=bool)
    if "test_mask" not in data:
        data.test_mask = torch.zeros(data.x.shape[0], dtype=bool)

    return data


def get_b(edge_index: PairTensor, weights: bool = True,
) -> Tensor:
    """
    Get normalized adjacency matrix B, defined as:

        B = I + D^{-1}A,

    where D is the degree matrix and A is the adjacency matrix.

    :param adj: Dense adjacency matrix.
    :param weights: Whether to consider edge weights.
    """
    adj = to_dense_adj(edge_index)[0]
    adj.fill_diagonal_(1)

    if not weights:
        adj[adj > 0] = 1

    if normalized:
        # adj /= adj.sum(axis=1).reshape(-1, 1)
        adj = torch.from_numpy(normalize(adj.numpy(), norm="l1", axis=0))

    return adj

def get_m(b: Adj, hops: int = 2):
    """
    Get t-hop node proximity matrix M, defined as:

        M = (B + B^2 + ... + B^t) / t,

    where B is the normalized adjacency matrix and t is the number of hops.

    :param edge_index: Edge index.
    :param hops: Number of hops.
    """
    return sum([torch.matrix_power(b, t) for t in range(1, hops + 1)]) / hops
