import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.typing import Tensor


def infer_node_time(data: Data, return_index: bool = False) -> Tensor:
    """
    Returns tensor with first interaction time for each node based on edge index and time.
    If an attribute `directed` is available, it will be used to filter out undirected edges.

    Example:

    >>> import torch
    >>> from torch_geometric.data import Data
    >>> data = Data(
            edge_index=torch.tensor([
                [1, 2, 4, 4, 6, 6],
                [0, 1, 3, 2, 5, 2]
            ]),
            time=torch.tensor([0, 0, 1, 1, 2, 3])
        )
    >>> infer_node_time(data)

    tensor([0, 0, 0, 1, 1, 2, 2])
    """
    edge_index = data.edge_index
    time = data.time

    # Edges with interaction time.
    e = np.column_stack([edge_index.t(), time.t()])
    e = e[np.argsort(e[:, -1])]
    e = e[np.unique(e[:, :2], axis=0, return_index=True)[1]]

    # Source nodes with interaction time.
    u = e[:, [0, -1]]
    u = u[u[:, -1].argsort()][:, [0, -1]]
    u = u[np.unique(u[:, 0], return_index=True)[1]]

    # Target nodes with interaction time.
    v = e[:, [1, -1]]
    v = v[v[:, -1].argsort()][:, [0, -1]]
    v = v[np.unique(v[:, 0], return_index=True)[1]]

    # Minimum interaction time for each node.
    t = np.concatenate([u, v], axis=0)
    t = t[t[:, -1].argsort()][:, [0, -1]]
    t = t[np.unique(t[:, 0], return_index=True)[1]]

    if return_index:
        return torch.from_numpy(t[:, -1]).t(), torch.from_numpy(t[:, 0]).t()

    assert t.shape[0] == e.max()+1,\
        f"Time ({t.shape[0]}) does not match number of nodes {(e.max() + 1)}."

    return torch.from_numpy(t[:, -1])


def describe_data(data: Data) -> dict:
    """
    Returns dictionary with basic statistics of a PyG data object.

    Converts the data object to a networkx graph and computes the number of nodes, edges,
    interactions, subgraphs (components), node features, target classes, and time intervals.

    Note: data converted to networkx from PyG is undirected.

    :param data: PyG data object.
    """
    G = to_networkx(data, to_undirected=False, to_multi=True).to_undirected()
    H = to_networkx(data, to_undirected=False, to_multi=False).to_undirected()

    x = "x" in data
    components = list(nx.connected_components(G))
    time = data.time.unique().shape[0] if "time" in data else 1
    interval = (data.time.min().item(), data.time.max().item()) if "time" in data else None

    return {
        "nodes": data.num_nodes,               # |V|
        "edges": H.size(),                     # |E|
        "interactions": G.size(),              # |\\mathcal{E}|
        "subgraphs": len(components),          # S
        "x": data.x.shape[1] if x else 0,      # d^v
        "y": data.y.unique().shape[0],         # y
        "t": time,                             # t
        "interval": interval,                  # t_{min}, t_{max}
    }
