import os
from pathlib import Path
from typing import Literal, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx, to_networkx


def load_dataset(
    root: str,
    name: str,
    features: Optional[str] = None,
    temporal: bool = True,
    directed: bool = False,
    normalize: bool = False,
    split: Literal["transductive", "inductive"] = None,
) -> Data:
    """ Return PyG data object. """
    path = f"{root}/{name}"

    x = None
    if features:
        x = torch.tensor(np.load(f"{path}/{features}"))

    edge_index = torch.tensor(np.load(f"{path}/edge_index.npy"))
    y = torch.tensor(np.load(f"{path}/y.npy"))

    time = (
        torch.tensor(np.load(f"{path}/time.npy"))
        if temporal and os.path.isfile(f"{path}/time.npy")
        else torch.zeros(edge_index.size(1))
    )

    data = Data(x=x, edge_index=edge_index, y=y, time=time)

    if split:
        # Load train, validation, and test masks.
        splits = np.load(f"{path}/splits.npz")
        data.train_mask = torch.tensor(splits["train_mask"])
        data.val_mask = torch.tensor(splits["val_mask"])
        data.test_mask = torch.tensor(splits["test_mask"])

    if directed:
        # Remove edges added by PyG when converting a NetworkX digraph.
        edge_directed = np.load(f"{path}/edge_directed.npy")
        data.edge_directed = torch.tensor(edge_directed, dtype=bool)
        data.edge_index = data.edge_index[:, edge_directed]
        data.time = data.time[edge_directed]
        data.train_mask = data.train_mask[edge_directed]
        data.val_mask = data.val_mask[edge_directed]
        data.test_mask = data.test_mask[edge_directed]

    if normalize:
        # Obtain training node features only, if applicable.
        if split:
            train_nodes = data.edge_index[:, data.train_mask].unique()
            val_nodes = data.edge_index[:, data.val_mask].unique()
            test_nodes = data.edge_index[:, data.test_mask].unique()

            if split == "inductive":
                train_nodes = train_nodes[~torch.isin(
                    train_nodes, torch.cat([val_nodes, test_nodes]).unique())]
                val_nodes = val_nodes[~torch.isin(
                    val_nodes, torch.cat([train_nodes, test_nodes]).unique())]
                test_nodes = test_nodes[~torch.isin(
                    test_nodes, torch.cat([train_nodes, val_nodes]).unique())]

        # Normalize features for unit mean and zero variance.
        x_train = data.x[train_nodes] if split else data.x
        data.x = (data.x - x_train.mean(dim=0)) / x_train.std(dim=0)

    return data


def load_graph(root: str, name: str, directed: bool = False) -> nx.Graph:
    """ Return NetworkX graph. """
    data = load_data(root=root, name=name, directed=directed)
    G = to_networkx(data, to_undirected=False, to_multi=True)
    return G.to_directed() if directed else G.to_undirected()


def convert_tgc(
    path: str,
    features: str = "feature.emb",
    labels: str = "node2label.txt",
    edges: Optional[str] = "edges.txt",
    output: Optional[str] = None,
) -> None:
    """ Convert a TGC dataset to numpy arrays. """
    name = os.path.basename(path)

    output = output or "."
    output = Path(output + "/" + name)
    os.makedirs(output, exist_ok=True)

    G = nx.read_edgelist(f"{path}/{edges or f'{name}.txt'}",
                         nodetype=int,
                         data=(("time", float),),
                         create_using=nx.MultiDiGraph())

    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))

    with open(f"{path}/{labels}", "r", encoding="utf8") as f:
        y = {
            int(line.split()[0]):
                int(line.split()[1])
            for line in f.readlines()
        }
    nx.set_node_attributes(G, y, "y")

    with open(f"{path}/{features}", "r", encoding="utf8") as f:
        x = {
            int(float(line.split(" ", 1)[0])):
                np.array(line.split()[1:], dtype=float)
            for line in f.readlines()[1:]
        }
    nx.set_node_attributes(G, x, "x")

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    G = nx.relabel_nodes(G, mapping)
    time = np.array([t for u, v, t in G.edges(data="time")])
    data = from_networkx(G.to_undirected())

    edges = set(
        (*e, t)
        for e, t in zip(G.edges(), time)
    )

    edge_directed = [
        (*e, t) in edges
        for e, t in zip(
            data.edge_index.t().numpy().tolist(),
            data.time.numpy()
        )
    ]

    np.save(f"{output}/y.npy", data.y.numpy())
    np.save(f"{output}/x.npy", data.x.numpy())
    np.save(f"{output}/edge_index.npy", data.edge_index.numpy())
    np.save(f"{output}/time.npy", data.time.numpy())
    np.save(f"{output}/edge_directed.npy", np.array(edge_directed, dtype=bool))


if __name__ == "__main__":
    from sys import argv
    convert_tgc(argv[1], features=argv[2], output=f"new_{argv[2]}")