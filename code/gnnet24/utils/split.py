from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.typing import Tensor, Tuple

from .utils import infer_node_time


def temporal_node_split(data: Data, split: str = "transductive") -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns **node-level** 'train_mask', 'val_mask', and 'test_mask' attributes.

    Requires the 'train_mask', 'val_mask', and 'test_mask' attributes to be
    present in the data object as edge-level masks for the filtering of nodes.

    :param data: PyG data object.
    :param split: Type of split. Default is 'transductive'.

        - If 'transductive", the split is based on an edge-level filter. Nodes
          may appear in multiple sets, provided they interact with different
          nodes at different times.

        - If 'inductive', the split is based on a node-level filter. Nodes are
          disjoint between sets and are sequentially filtered by training,
          validation, and test masks.

        - If 'temporal', the split is based on a time-level filter. Nodes are
          filtered based on their first interaction time and are disjoint
          between sets, with future nodes appearing in the val/test sets.
    """
    assert split in ("transductive", "inductive", "temporal")

    # Transductive split.
    train_nodes = data.edge_index[:, data.train_mask].unique()
    val_nodes = data.edge_index[:, data.val_mask].unique()
    test_nodes = data.edge_index[:, data.test_mask].unique()

    if split == "inductive":
        test_nodes = test_nodes[~torch.isin(
            test_nodes, torch.cat([train_nodes, val_nodes]).unique())]
        val_nodes = val_nodes[~torch.isin(
            val_nodes, torch.cat([train_nodes, test_nodes]).unique())]
        train_nodes = train_nodes[~torch.isin(
            train_nodes, torch.cat([val_nodes, test_nodes]).unique())]

    if split == "temporal":
        assert "time" in data, "Temporal split requires 'time' attribute."
        # Get last training time and first test time.
        time_train = data.time[data.train_mask].max()
        time_test = data.time[data.test_mask].min()
        # Infer node time if not available.
        node_time = data.time if "time" in data.node_attrs() else infer_node_time(data)
        # Get nodes in training, validation, and test intervals.
        node_index = torch.arange(data.num_nodes)
        train_nodes = node_index[node_time <= time_train]
        val_nodes = node_index[(node_time > time_train) & (node_time < time_test)]
        test_nodes = node_index[node_time >= time_test]

    train_mask = torch.zeros(data.num_nodes, dtype=bool)
    val_mask = torch.zeros(data.num_nodes, dtype=bool)
    test_mask = torch.zeros(data.num_nodes, dtype=bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    assert train_mask.sum() > 0, "No training nodes."
    assert test_mask.sum() > 0, "No test nodes."
    return train_mask, val_mask, test_mask


def temporal_edge_split(data: Data, level: str, train_ratio: float, val_ratio: Optional[float] = None):
    """
    Returns **edge-level** 'train_mask', 'val_mask', and 'test_mask' attributes.

    Training, validation, and test sets are selected based on the number of
    nodes in each (sequential and disjoint) time interval, with the validation
    set corresponding to the interval between the training and test sets.

    :param data: PyTorch Geometric object.
    :param level: Level of split. Can be 'node' or 'edge'.
        Edge-level masks are returned in both cases.
        - If 'node', the split is based on the number of nodes in each time
          interval.
        - If 'edge', the split is based on the number of edges in each time
          interval.
    :param train_ratio: Proportion of training nodes.
    :param val_ratio: Proportion of validation nodes. Optional.
    """
    assert 0 < train_ratio < 1, "Training ratio must be between 0 and 1."
    assert level in ("node", "edge"), "Level must be 'node' or 'edge'."
    assert "time" in data, f"Attribute 'time' not found in data object."

    # NOTE: PyG does not filter nodes on `edge_subgraph`.
    subgraph = data.subgraph if "time" in data.node_attrs() else data.edge_subgraph
    val_mask = torch.zeros(data.num_edges, dtype=bool)

    for train_time in data.time.unique().numpy()[::-1]:
        train_mask = data.time <= train_time
        s = subgraph(train_mask)

        if train_ratio >= (
            s.edge_index.unique().shape[0]/data.num_nodes
            if level == "node"
            else s.num_edges/data.num_edges
        ):
            break

    if val_ratio:
        for val_time in range(train_time, data.time.max()+1)[::-1]:
            val_mask = (data.time > train_time) & (data.time <= val_time)
            s = subgraph(val_mask)

            if val_mask.sum() == 0 and val_time < data.time.max():
                val_mask = prev_mask
                break

            if val_ratio >= (
                s.edge_index.unique().shape[0]/data.num_nodes
                if level == "node"
                else s.num_edges/data.num_edges
            ):
                break
            prev_mask = val_mask

    test_mask = ~(train_mask|val_mask)
    assert train_mask.sum() > 0, "Training set is empty."
    assert not val_ratio or val_mask.sum() > 0, "Validation set is empty."
    assert test_mask.sum() > 0, "Test set is empty."

    return train_mask, val_mask, test_mask
