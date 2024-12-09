import os
import os.path as osp
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx


def load_embeddings(filepath: str, arr_key: str = "arr_0") -> torch.Tensor:
    """
    Load node embeddings from disk.

    Accepted file formats:
        - '.npy': NumPy array.
        - '.npz': NumPy compressed array.
        - '.emb': Node2Vec embeddings.

    :param filepath: Path to file.
    :param arr_key: Key to extract from '.npz' file. Default is 'arr_0'.
    """
    name, ext = osp.splitext(filepath)
    ext = ext or (
        ".npy"
        if osp.exists(f"{name}.npy")
        else ".npz"
        if osp.exists(f"{name}.npz")
        else ".emb"
        if osp.exists(f"{name}.emb")
        else None
    )

    assert ext in (".npy", ".npz", ".emb"),\
        "Invalid file format, expected '.npy', '.npz', or '.emb'."

    if ext == ".npy":
        return np.load(f"{filepath}.npy")

    if ext == ".npz":
        return np.load(f"{filepath}.npz")[arr_key]

    if ext == ".emb":
        with open(f"{filepath}.emb", "r", encoding="utf8") as f:
            x = {
                int(line.split()[0]):
                    list(map(float, line.split()[1:]))
                for line in f.readlines()[1:]
            }
        return np.array([x[i] for i in range(len(x))])


def load_tgc_data(
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
