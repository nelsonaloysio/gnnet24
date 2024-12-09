import logging as log
import os.path as osp
from typing import Literal, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from ..utils import (
    temporal_edge_split,
    temporal_node_split,
    load_embeddings,
)


def transform(
    data: Data,
    pretrained_features: str = None,
    temporal: bool = True,
    normalized: bool = False,
    discretized: bool = False,
    split: Optional[Literal["transductive", "inductive", "temporal"]] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    level_ratio: Optional[Literal["node", "edge"]] = "node",
) -> Data:
    """
    :param pretrained_features: Choice of pretrained node features to load
        among: ('node2vec', 'attri2vec', 'dynnode2vec', 'tnodeembed').
        If None (default), load original features, if available.
    :param temporal: Whether to add edge-level 'time' attribute. Default is True.
    :param normalized: Whether to normalized features by subtracting the mean
        and dividing by the standard deviation. Default is False.
    :param discretized: Whether to discretize time stamps. Default is False.
    :param split: Whether to perfor ma node-level split for train/val/test sets.
        Choices: ('transductive', 'inductive', 'temporal'). Default is None.
    :param level_ratio: Whether train_ratio and val_ratio are node-level or
        edge-level. Choices: ('node', 'edge'). Default is 'node'.
    :param train_ratio: Proportion of training nodes. Optional.
    :param val_ratio: Proportion of validation nodes. Optional.
    """
    assert level_ratio in ("node", "edge"), "Split level must be either 'node' or 'edge'."
    assert temporal or split != "temporal", "Temporal split requires `temporal=True`."
    assert discretized or split != "temporal", "Temporal split requires `discretized=True`."

    # Load pretrained node features.
    if pretrained_features:
        data.x = torch.from_numpy(load_embeddings(pretrained_features))

    # Factorize time stamps as integer values.
    # NOTE: does not keep distance between time stamps.
    if discretized:
        data.time = torch.from_numpy(
            np.unique(data.time.numpy(), return_inverse=True)[1])

    # Split nodes into train/val/test sets.
    if split:
        # Obtain edge-level train/val/test masks.
        if train_ratio:
            data.train_mask, data.val_mask, data.test_mask = (
                temporal_node_split(data, train_ratio, val_ratio)
                if level_ratio == "node"
                else temporal_edge_split(data, train_ratio, val_ratio)
            )
        data.train_mask, data.val_mask, data.test_mask = temporal_node_split(data, split)
        masks = [getattr(data, f'{m}_mask').sum().item() for m in ("train", "val", "test")]
        pct = map(lambda x: f"{x/data.x.shape[0]:.2f}", masks)
        log.info("Split '%s': %s/%s/%s (%s/%s/%s) train/val/test.", split, *masks, *pct)
    else:
        # Create train/val/test masks on full graph.
        data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.ones(data.num_nodes, dtype=torch.bool)

    # Normalize node features.
    if normalized:
        x_norm = data.x[data.train_mask] if split else data.x
        data.x = (data.x - x_norm.mean(dim=0)) / x_norm.std(dim=0)

    # Static graph (t=0 for all edges).
    if not temporal:
        data.time = torch.zeros(data.edge_index.shape[1])

    return data