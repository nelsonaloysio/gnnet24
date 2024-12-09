import logging as log
from typing import Literal, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from .split import (
    temporal_edge_split,
    temporal_node_split,
)
from .utils import load_embeddings


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
        among: ('node2vec', 'dynnode2vec', 'tnodeembed').
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

    if split:
        # Obtain edge-level train/val/test masks.
        if train_ratio:
            train_mask, val_mask, test_mask = temporal_edge_split(
                data, train_ratio, val_ratio)
        # Split nodes into train/val/test sets.
        train_mask, val_mask, test_mask = temporal_node_split(data, split)
        masks = [getattr(data, f'{m}_mask').sum().item() for m in ("train", "val", "test")]
        pct = map(lambda x: f"{x/data.x.shape[0]:.2f}", masks)
        log.info("Split '%s': %s/%s/%s (%s/%s/%s) train/val/test.", split, *masks, *pct)
    else:
        # Create train/val/test masks on full graph.
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        val_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        test_mask = torch.ones(data.num_nodes, dtype=torch.bool)

    # Normalize node features.
    if normalized:
        x_norm = data.x[train_mask] if split else data.x
        data.x = (data.x - x_norm.mean(dim=0)) / x_norm.std(dim=0)

    # Static graph (t=0 for all edges).
    if not temporal:
        data.time = torch.zeros(data.edge_index.shape[1])

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
