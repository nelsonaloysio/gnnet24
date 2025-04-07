import os
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import scipy
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils.convert import from_networkx

from .utils import load_embeddings


def from_tgc(
    path: str,
    x_filename: str = "feature.emb",
    y_filename: str = "node2label.txt",
    edge_index: Optional[str] = "edges.txt",
    output: Optional[str] = None,
) -> None:
    """ Convert a TGC dataset to numpy arrays. """
    name = Path(path).basename()

    output = output or "."
    output = Path(output).joinpath(name)
    os.makedirs(output, exist_ok=True)

    # Load edge list with time stamps.
    G = nx.read_edgelist(f"{path}/{edge_index or f'{name}.txt'}",
                         nodetype=int,
                         data=(("time", float),),
                         create_using=nx.MultiDiGraph())

    # Remove self-loops and isolated nodes.
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))

    # Load node labels.
    with open(f"{path}/{y_filename}", "r", encoding="utf8") as f:
        y = {
            int(line.split()[0]):
                int(line.split()[1])
            for line in f.readlines()
        }
    nx.set_node_attributes(G, y, "y")

    # Load node features.
    if x_filename:
        x = load_embeddings(f"{path}/{x_filename}")
        nx.set_node_attributes(G, x, "x")

    # Relabel nodes to consecutive integers.
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    G = nx.relabel_nodes(G, mapping)

    # Convert to PyG data object.
    data = from_networkx(G.to_undirected())

    # Ensure time starts at zero if all values are positive.
    if "time" not in data:
        data.time = np.ones(G.order())
    if "time" in data and data.time.min() > 0:
        data.time -= data.time.min()

    return data


def to_dmon(data: Data) -> dict:
    """
    Converts a PyG data object for compatibility with DMoN.

    :param data: PyG data object.
    """
    csr_feature = scipy.sparse.csr_matrix(data.x)
    csr_adj = to_scipy_sparse_matrix(data.edge_index).tocsr()

    return dict(
        labels=data.y.numpy(),
        label_indices=np.arange(data.num_nodes),
        feature_data=csr_feature.data,
        feature_indices=csr_feature.indices,
        feature_indptr=csr_feature.indptr,
        feature_shape=csr_feature._shape,
        adj_data=csr_adj.data,
        adj_indices=csr_adj.indices,
        adj_indptr=csr_adj.indptr,
        adj_shape=csr_adj._shape,
        # time=...,
    )