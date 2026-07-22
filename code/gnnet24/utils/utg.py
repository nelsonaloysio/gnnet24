#!/usr/bin/env python3

import logging as log
import os
from time import time
from typing import Literal, Optional

import networkx as nx
import networkx_temporal as tx
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_networkx, from_networkx
from torch.utils.data import Dataset


def get_utg_node_index(data: Dataset)-> list:
    """
    Returns temporal node indices, i.e., a list of lists
    with the indices of each node in the temporal node index.

    Passing a `mask` allows to obtain a temporal node index
    considering only unmasked nodes, thus changing the length,
    as well as the resulting indices for each item in the list.

    Return an empty list if `data` has no `node_index` or 'time'.

    :param data: PyTorch Geometric data object.
    :param node_index: 1-dimensional tensor
    :param mask: 1-dimensional tensor
    """
    time = getattr(data, "time")
    node_index = getattr(data, "node_index")

    if node_index is None or time is None:
        return []

    # temporal_node_index = {}
    # for i, node in enumerate(node_index if mask is None else node_index[mask]):
    #     temporal_node_index[int(node)] = temporal_node_index.get(int(node), []) + [i]
    # return list(temporal_node_index.values())

    temporal_node_index = [[] for _ in range(node_index.max() + 1)]
    for i, node in enumerate(node_index):
        temporal_node_index[int(node)] += [time[i]]

    return temporal_node_index


def get_utg_array(
    array: ndarray,
    temporal_node_index: list,
    how: Literal["first", "last", "average"] = "first",
    mask: Optional[ndarray] = None,
) -> ndarray:
    """
    Obtain array from temporal nodes, i.e., a new array
    with the same features from the `temporal_node_index`.
    If `temporal_node_index` is empty, return `array`.

    Passing a `mask` allows to filter the input `array` and
    avoid the need of filtering the output `temporal_array`.
    The end result is the same as the following snippet:

    >>> temporal_mask = get_utg_array(mask, temporal_node_index)
    >>> temporal_array = get_utg_array(array, temporal_node_index)[temporal_mask]

    For tensors, use `get_utg_tensor` instead.

    :param array: array, shape (n, d)
    :param temporal_node_index: list
    :param how: str in ('first', 'last', 'average')
    :param mask: array, shape (n, 1)
    """
    assert how in ("first", "last", "average"),\
           f"Argument 'how' must be in ('first', 'last', 'average')."

    if not temporal_node_index:
        return array

    return np.array([
        np.average(array[idx], axis=0) if how == "average" else\
        array[idx[-1]] if how == "last" else\
        array[idx[0]]
        for idx in temporal_node_index
        if mask is None or mask[idx[0]]
    ])


def get_utg_tensor(
    tensor: Tensor,
    temporal_node_index: list,
    how: Literal["first", "last", "average"] = "first",
    mask: OptTensor = None,
) -> Tensor:
    """
    Obtain tensor from temporal nodes, i.e., a new tensor
    with the same features from the `temporal_node_index`.
    If `temporal_node_index` is empty, return `tensor`.

    Passing a `mask` allows to filter the input `tensor` and
    avoid the need of filtering the output `temporal_array`.
    The end result is the same as the following snippet:

    >>> temporal_mask = get_utg_tensor(mask, temporal_node_index)
    >>> temporal_array = get_utg_tensor(array, temporal_node_index)[temporal_mask]

    For numpy arrays, use `get_utg_array` instead.

    :param tensor: tensor, shape (n, d)
    :param temporal_node_index: list
    :param how: str in ('first', 'last', 'average')
    :param mask: array, shape (n, 1)
    """
    assert how in ("first", "last", "average"),\
           f"Argument 'how' must be in ('first', 'last', 'average')."

    if not temporal_node_index:
        return tensor

    return torch.stack([
        tensor[idx].mean(axis=0) if how == "average" else\
        tensor[idx[-1]] if how == "last" else\
        tensor[idx[0]]
        for idx in temporal_node_index
        if mask is None or mask[idx[0]]
    ])


def to_utg(
    data: Dataset,
    bins: int,
    setting: Literal["transductive", "inductive"] = "transductive",
    output: str = ".",
    verify_pubmed: bool = False
) -> Dataset:
    """
    Build the Unified Temporal Graph (UTG) data.

    :param bins: int
        The number of bins to slice the temporal graph.
    :param setting: str
        Learning setting, either 'transductive' or 'inductive'.
    :param output: str
        The output folder path to save graphs and logs to.
    :param verify_pubmed: bool
        Whether to verify the PubMed data.
    """
    assert type(bins) in (int, list, range),\
           f"Bins must be an integer or a list of integers."

    assert setting in ("transductive", "inductive"),\
           f"Setting must be either 'transductive' or 'inductive'."

    bins = [bins] if type(bins) == int else bins

    os.makedirs(output, exist_ok=True)

    name = data.name if hasattr(data, "name") else data.__class__.__name__.lower()

    # Building graph using torch_geometric resulted in undirected edges,
    # even with `to_undirected=False` or when calling `to_directed()` after.
    G = to_networkx(
        data[0],
        node_attrs=["x", "y", "t"],
        to_undirected=True
    )\
    .to_directed()

    # Obtain edges from graph.
    edges = nx.to_pandas_edgelist(G)
    log.info(f"{G}")

    # Obtain train, validation, and test subgraphs.
    if setting == "transductive":
        # Overlapping sets in which only the source node (papers citing others) is considered for the train/val/test set.
        G_train = G.edge_subgraph(edges[edges["source"].apply(lambda x: bool(data.train_mask[x]))].apply(tuple, axis=1))
        G_val = G.edge_subgraph(edges[edges["source"].apply(lambda x: bool(data.val_mask[x]))].apply(tuple, axis=1))
        G_test = G.edge_subgraph(edges[edges["source"].apply(lambda x: bool(data.test_mask[x]))].apply(tuple, axis=1))

    elif setting == "inductive":
        # Disjoint sets in which both source and target nodes must be in the specific train/val/test set.
        G_train = G.edge_subgraph(edges[edges.map(lambda x: bool(data.train_mask[x])).all(axis=1)].apply(tuple, axis=1))
        G_val = G.edge_subgraph(edges[edges.map(lambda x: bool(data.val_mask[x])).all(axis=1)].apply(tuple, axis=1))
        G_test = G.edge_subgraph(edges[edges.map(lambda x: bool(data.test_mask[x])).all(axis=1)].apply(tuple, axis=1))

    log.info(f"G: {G}")
    log.info(f"G_train: {G_train}")
    log.info(f"G_val: {G_val}")
    log.info(f"G_test: {G_test}")

    # Snapshot-based Temporal Graph (STG)
    # Obtain snapshot-based temporal graphs from train, validation, and test subgraphs.
    log.info(f"Building STG with {bins} bins...")
    slice_params = dict(bins=bins, attr="t", attr_level="node", node_level="source")
    t0 = time()

    STG_train = tx.from_static(G_train).slice(qcut=True, **slice_params)
    if len(STG_train) != bins:
        log.warning(f"STG_train resulted in {len(STG_train)} bins, but expected {bins}. Recreating with qcut=False...")
        STG_train = tx.from_static(G_train).slice(qcut=False, **slice_params)

    STG_val = tx.from_static(G_val).slice(qcut=True, **slice_params)
    if len(STG_val) != bins:
        log.warning(f"STG_val resulted in {len(STG_val)} bins, but expected {bins}. Recreating with qcut=False...")
        STG_val = tx.from_static(G_val).slice(qcut=False, **slice_params)

    STG_test = tx.from_static(G_test).slice(qcut=True, **slice_params)
    if len(STG_test) != bins:
        log.warning(f"STG_test resulted in {len(STG_test)} bins, but expected {bins}. Recreating with qcut=False...")
        STG_test = tx.from_static(G_test).slice(qcut=False, **slice_params)

    # Obtain snapshot-based temporal graph `STG` from the train, validation, and test snapshots.
    STG = tx.from_snapshots(STG_train.to_snapshots() + STG_val.to_snapshots() + STG_test.to_snapshots())

    log.info(STG)
    list(log.info(f"STG_train_t={t}: {S}") for t, S in enumerate(STG_train))
    list(log.info(f"STG_val_t={t}: {S}") for t, S in enumerate(STG_val))
    list(log.info(f"STG_test_t={t}: {S}") for t, S in enumerate(STG_test))

    # Relabel nodes to avoid proxying training nodes in validation and test snapshots.
    relabel_nodes = [
        {}
        if t in range(len(STG_train)) else
        {v: f"{v}_{next(iter((STG.temporal_index(v))))}" for v in STG[t].nodes() if data.train_mask[v]}
        for t in range(len(STG))
    ]

    # Convert `STG` into a unified temporal graph `UTG`.
    log.info(f"Building UTG from {len(STG)} snapshots...")

    UTG = STG.to_unified(add_couplings=True,
                        relabel_nodes=relabel_nodes,
                        node_index=G.nodes())

    UTG.name = f"{name}_utg_{setting[:1].upper()}_t={bins}"
    # UTG.name = f"{name}_utg_{setting[:1].upper()}_t={len(STG)}"

    log.info(f"{UTG} ("
            f"train: {data.train_mask.sum()/data.x.shape[0]:.2f}, "
            f"val: {data.val_mask.sum()/data.x.shape[0]:.2f}, "
            f"test: {data.test_mask.sum()/data.x.shape[0]:.2f})")

    # Convert `UTG` into a PyTorch Geometric data.
    # We here consider the graph as undirected to match Planetoid,
    # but we could also reverse edges during training instead:
    # #2915: https://github.com/pyg-team/pytorch_geometric/issues/2915#issuecomment-889207543
    # #3735: https://github.com/pyg-team/pytorch_geometric/issues/3735#issuecomment-998622047
    UTG_data = from_networkx(UTG.to_undirected())

    # Create train, validation, and test masks.
    UTG_data.train_mask = torch.from_numpy(np.array([True if data.train_mask[v] else False for v in UTG_data.node_index]))
    UTG_data.test_mask = torch.from_numpy(np.array([True if data.test_mask[v] else False for v in UTG_data.node_index]))
    UTG_data.val_mask = torch.from_numpy(np.array([True if data.val_mask[v] else False for v in UTG_data.node_index]))

    # Save data to disk.
    log.info(f"Finished in {time()-t0:.3f}s....")

    if output:
        log.info(f"Writing to '{output}'...")
        torch.save(UTG_data, f"{output}/{UTG_data.name}.pt")

        # Compare the original graph, its snapshots, and the unified graph
        df = pd.DataFrame({
            "G": {"V": G.order(), "E": G.size()},
            **{f"STG_train_t={t}": {"V": f"{STG_train[t].order()}", "E": f"{STG_train[t].size()}"} for t in range(len(STG_train))},
            **{f"STG_val_t={t}": {"V": f"{STG_val[t].order()}", "E": f"{STG_val[t].size()}"} for t in range(len(STG_val))},
            **{f"STG_test_t={t}": {"V": f"{STG_test[t].order()}", "E": f"{STG_test[t].size()}"} for t in range(len(STG_test))},
            "UTG": {"V": UTG.order(), "E": UTG.size()},
        })\
        .fillna("")\
        .transpose()

        df.to_csv(f"{output}/graphs.csv")
        df.to_markdown(f"{output}/graphs.md")
        df.to_latex(f"{output}/graphs.tex")

    if verify_pubmed:
        assert name.lower() == "pubmed", f"Dataset to verify must be 'PubMed'."

        # Verify unified data matches expectations.
        log.info("Verifying data...")

        # No isolates should be found.
        log.info(f"Isolates: {list(nx.isolates(UTG))}")

        for v in (8998, 8945, 16626):
            # Validation and test graphs should have non-proxied training nodes,
            # e.g., for Planetoid PubMed, node `8998`, with rewired edges.
            log.info(f"Node {v} (t={dict(G.nodes(data='t'))[v]}) is in "
                    f"'{next(iter([mask for mask in ['train', 'val', 'test'] if getattr(data, f'{mask}_mask')[v]]))}' mask")
            # If transductive, sets should be overlapping due to edges connecting,
            # e.g., for Planetoid PubMed, nodes in train/test sets to node `8945` in validation set.
            log.info(f"Node {v} is in snapshots {STG.temporal_index(v)}")
            # If inductive, sets should be disjoint due to removed edges,
            # e.g., for Planetoid PubMed, node `16626` is removed due to not having edges to other test nodes.
            log.info(f"Node {v} is in UTG as {[t for t in range(len(STG)) if UTG.has_node(f'{v}_{t}')]}")

    return UTG_data
