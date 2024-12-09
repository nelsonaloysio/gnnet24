from functools import reduce
from operator import add
from typing import Literal

import numpy as np
from numpy import ndarray
from torch import Tensor
from torch_geometric.typing import Adj


def get_edge_mask(
    edge_index: Tensor,
    node_mask: ndarray,
    how: Literal["source", "target", "source+target"] = "source"
) -> ndarray:
    """
    Returns edge mask from node mask.

    :param edge_index: Edge index tensor.
    :param node_mask: Node mask array.
    :param how: Method to mask edges.
        - "source": mask edges with source nodes in node mask.
        - "target": mask edges with target nodes in node mask.
        - "source+target": mask edges with source and target nodes in node mask.
    """
    assert how in ["source", "target", "source+target"],\
            f"Argument `how` must be in ('source', 'target', 'source+target')."

    if how == "source":
        return np.array([True if node_mask[u] else False for u, v in zip(*edge_index)])

    if how == "target":
        return np.array([True if node_mask[v] else False for u, v in zip(*edge_index)])

    return np.array([True if node_mask[u] and node_mask[v] else False for u, v in zip(*edge_index)])


def get_submask(*masks) -> ndarray:
    """
    Returns submask, with booleans refering to
    and with the same length as the last mask.

    :param masks: (n, 1) array(s).
    """
    assert len(masks) == 2,\
           "Not implemented for >2 masks."

    mask = masks[0]
    array = np.array(range(len(mask)))

    for i in range(1, len(masks)):
        array = np.array([True if masks[i][a] else False for n, a in enumerate(array[mask+masks[i]])])

    return array


def mask_adj(adj: Adj, node_mask: ndarray) -> Tensor:
    """
    Returns masked adjacency tensor.

    :param adj: Adjacency matrix.
    :param node_mask: Node mask array.
    """
    return adj[:, node_mask][node_mask]


def mask_edge_index(edge_index: Tensor, edge_mask: ndarray) -> Tensor:
    """
    Returns masked edge index tensor.

    :param edge_index: Edge index tensor.
    :param edge_mask: Edge mask array.
    """
    return edge_index[:, edge_mask]


def submask_array(array: ndarray, *masks) -> ndarray:
    """
    Returns array, sequentially masked by the given masks.

    :param array: (n, d) array.
    :param masks: (n, 1) array(s).
    """
    assert len(masks) == 2,\
           "Not implemented for >2 masks."

    array = array[reduce(add, masks)]

    for i in range(1, len(masks)):
        array = array[get_submask(*masks[i-1:i+1])]

    return array
