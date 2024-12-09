from collections import Counter
from os import makedirs
from typing import Literal, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.convert import from_networkx, to_networkx

# TODO: Remove line when temopral PubMed is available from Planetoid.
from pubmed_temporal.build import Planetoid as Planetoid_

from .mask import get_edge_mask


def tgc_from_nx(G: nx.Graph, name: str, time_attr: str = "time"):
    makedirs(name, exist_ok=True)

    # Remove self-loops.
    G.remove_edges_from(nx.selfloop_edges(G))

    # Remove isolates.
    G.remove_nodes_from(list(nx.isolates(G)))

    # Obtain temporal data.
    time = None
    if next(iter(G.edges(data=time_attr)))[-1] != None:
        time = True
    if next(iter(G.nodes(data=time_attr)))[-1] != None:
        time = dict(G.nodes(data=time_attr))

    with open(f"{name}/feature.txt", "w") as fa:
        list(fa.write(f"{x[-1]}\n".strip('[]').replace(',', '').replace('[', '').replace(']', '')) for x in G.nodes(data="x"))
        with open(f"{name}/feature.emb", "w") as fb:
            fb.write(f"{G.order()} {len(next(iter(G.nodes(data=True)))[-1])}\n")
            list(fb.write(f"{x[0]} {x[-1]}\n".strip('[]').replace(',', '').replace('[', '').replace(']', '')) for x in G.nodes(data="x"))

    with open(f"{name}/node2label.txt", "w") as f:
        list(f.write(f"{x[0]} {x[-1]}\n".strip('[]').replace(',', '').replace('[', '').replace(']', '')) for x in G.nodes(data="y"))

    with open(f"{name}/edges.txt", "w") as f:
        list(
            f.write(
                f"{e[0]} {e[1]} 0\n"
                if time is None else
                f"{e[0]} {e[1]} {e[-1]}\n"
                if time is True else
                f"{e[0]} {e[1]} {time[e[0]]}\n"
            )
        for e in G.edges(data=time_attr if time is True else False))


def tgc_from_torch(data, name: str = None, time_attr: str = "time"):
    if name is None:
        name = getattr(data, "name", data.__class__.name)

    makedirs(name, exist_ok=True)

    # Obtain temporal data.
    time = None
    if time_attr in data:
        time = getattr(data, time_attr)
        if data.is_node_attr(time_attr):
            time = time[data.edge_index[0]]

    # Remove self-loops.
    data.edge_index, mask = remove_self_loops(data.edge_index, edge_attr=time)
    if time is not None:
        setattr(data, time_attr, time[mask])

    # Remove isolates.
    data = RemoveIsolatedNodes().forward(data)

    with open(f"{name}/feature.emb", "w") as fb:
        fb.write(f"{data.x.shape[0]} {data.x.shape[1]}\n")
        list(fb.write(f"{i} {x}\n".strip('[]').replace(',', '').replace('[', '').replace(']', '')) for i, x in enumerate(data.x.numpy().tolist()))

    with open(f"{name}/node2label.txt", "w") as f:
        list(f.write(f"{i} {y}\n".strip('[]').replace(',', '')) for i, y in enumerate(data.y.numpy().tolist()))

    with open(f"{name}/edges.txt", "w") as f:
        list(
            f.write(
                (
                    f"{e[0]} {e[1]} 0\n"
                    if time is None else
                    f"{e[0]} {e[1]} {time[i]}\n"
                )
            )
            for i, e in enumerate(data.edge_index.t().numpy().tolist())
        )


def citeseer(path: str) -> nx.DiGraph:
    G = nx.DiGraph()

    with open(f"{path}/citeseer.cites", "r") as f:
        edges = [line.strip().split() for line in f.readlines()]

    with open(f"{path}/citeseer.content", "r") as f:
        lines = [line.strip().split() for line in f.readlines()]
        nodes = pd.DataFrame(
            {
                "x": [[int(x) for x in line[1:-1]] for line in lines],
                "y": [line[-1] for line in lines],
            },
            index=[line[0] for line in lines],
        )

    # Add nodes and edges, reversing its direction.
    G.add_nodes_from(nodes.index)
    G.add_edges_from([(v, u) for u, v in edges])

    # Add features `x`, corresponding to the original binary feature vectors.
    nx.set_node_attributes(G, values=nodes["x"], name="x")

    # Add classes `y`, mapping them to match Planetoid's.
    y_map = {"AI": 0, "ML": 1, "IR": 2, "DB": 3, "Agents": 4, "HCI": 5}
    nx.set_node_attributes(G, values=nodes["y"].map(y_map.get).astype(int), name="y")

    # Remove nodes without features.
    nodes_without_features = [k for k, v in Counter(nodes.index.tolist() + list(G.nodes())).items() if v == 1]
    G.remove_nodes_from(nodes_without_features)

    # Remove self-loops.
    G.remove_edges_from(nx.selfloop_edges(G))

    # Remove isolates.
    G.remove_nodes_from(list(nx.isolates(G)))

    # Relabel nodes from str to int.
    nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes)}, copy=False)

    # Write to files.
    tgc_from_nx(G, "citeseer")


def cora(path: str) -> nx.DiGraph:
    G = nx.DiGraph()

    with open(f"{path}/cora.cites", "r") as f:
        edges = [line.strip().split() for line in f.readlines()]

    with open(f"{path}/cora.content", "r") as f:
        lines = [line.strip().split() for line in f.readlines()]
        nodes = pd.DataFrame(
            {
                "x": [[int(x) for x in line[1:-1]] for line in lines],
                "y": [line[-1] for line in lines],
            },
            index=[line[0] for line in lines],
        )

    # Reverse edge direction.
    G.add_edges_from([(v, u) for u, v in edges])

    # Add features `x`, corresponding to the original binary feature vectors.
    nx.set_node_attributes(G, values=nodes["x"], name="x")

    # Add classes `y`, mapping them to match Planetoid's.
    y_map = {"Theory": 0, "Reinforcement_Learning": 1, "Genetic_Algorithms": 2, "Neural_Networks": 3, "Probabilistic_Methods": 4, "Case_Based": 5, "Rule_Learning": 6}
    nx.set_node_attributes(G, values=nodes["y"].map(y_map.get).astype(int), name="y")

    # Remove self-loops.
    G.remove_edges_from(nx.selfloop_edges(G))

    # Relabel nodes from str to int.
    nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes)}, copy=False)

    # Write to files.
    tgc_from_nx(G, "cora")


def karateclub() -> nx.DiGraph:
    data = KarateClub()[0]
    tgc_from_torch(data, "karateclub")


def pubmed(learning_setting: Optional[Literal["transductive", "inductive"]]) -> nx.DiGraph:
    if learning_setting is None:
        data = Planetoid_(name="PubMed", split="temporal")[0]
        tgc_from_torch(data, "pubmed")

    elif learning_setting in ("transductive", "inductive"):

        for m in ["train_mask", "val_mask", "test_mask"]:
            data = Planetoid_(name="PubMed", split="temporal")[0]
            # Filter edges by direction
            edge_directed = data.pop("edge_directed")
            data.edge_index = data.edge_index[:, edge_directed]
            data.edge_time = data.edge_time[edge_directed]
            # Filter edges by split
            mask = getattr(data, m)
            edge_mask = get_edge_mask(data.edge_index, mask, "source+target" if learning_setting == "inductive" else "source")
            data.edge_index = data.edge_index[:, edge_mask]
            data.edge_time = data.edge_time[edge_mask]
            # Relabel source and target nodes
            nodes = data.edge_index.view(-1).unique().numpy()
            relabel = {node: i for i, node in enumerate(nodes)}
            data.edge_index = torch.tensor([relabel[node] for node in data.edge_index.view(-1).numpy()]).view(2, -1)
            node_mask = torch.zeros(data.num_nodes, dtype=bool)
            node_mask[nodes] = True
            # Filter nodes by split
            data.x = data.x[node_mask]
            data.y = data.y[node_mask]
            data.time = data.time[node_mask]
            data.train_mask = data.train_mask[node_mask]
            data.val_mask = data.val_mask[node_mask]
            data.test_mask = data.test_mask[node_mask]
            # Write to files
            tgc_from_torch(data, f"pubmed-{learning_setting}/{m.split('_')[0]}", time_attr="edge_time")
            with open(f"pubmed-{learning_setting}/{m.split('_')[0]}/nodes.txt", "w") as f:
                list(f.write(f"{node}\n") for node in nodes)
            # Filter n2v embeddings
            # H = {}
            # with open("pubmed/feature_n2v.emb", "r") as f:
            #     total, dim = f.readline().strip().split()
            #     for line in f.readlines():
            #         node, h = line.strip().split(" ", 1)
            #         H[int(node)] = h
            #     with open(f"pubmed-{learning_setting}/{m.split('_')[0]}/nodes.txt", "w") as f:
            #         nodes = [int(_.strip()) for _ in f.readlines()]
            #     nodes = {node: i for i, node in enumerate(sorted(nodes))}
            #     print(nodes)
            #     with open(f"pubmed-{learning_setting}/{m.split('_')[0]}/feature_n2v.emb", "w") as f:
            #         f.write(f"{len(nodes)} {dim}\n")
            #         for node, h in H.items():
            #             if node in nodes:
            #                 f.write(f"{nodes[node]} {h}\n")

    else:
        raise ValueError(f"Invalid learning setting, expected 'transductive' or 'inductive'.")
