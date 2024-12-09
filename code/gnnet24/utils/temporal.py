from copy import copy
from typing import Optional, Union
from typing_extensions import Self

import torch
import torch_geometric as pyg
from torch.utils.data import Dataset
from torch_geometric.utils import select


# [pyg-team/pytorch_geometric#9333]
def snapshot(
    data: Dataset,
    start_time: Union[float, int],
    end_time: Union[float, int],
    attr: Optional[str] = "time",
) -> Self:
    out = copy(data)

    for store in out.stores:
        mask = getattr(store, attr)
        mask = (mask >= start_time) & (mask <= end_time)

        # num_nodes = pyg.utils.num_nodes.maybe_num_nodes(
        #         data.edge_index, getattr(data, "num_nodes", None))

        if store.is_node_attr(attr):
            keys = store.node_attrs()
            num_nodes = int(mask.sum())
        elif store.is_edge_attr(attr):
            keys = store.edge_attrs()
            num_nodes = int(mask.sum())

        for key, dim in store._cat_dims(keys).items():
            store[key] = select(store[key], mask, dim)

        if store.is_node_attr(attr):
            keys = store.edge_attrs()
            vals = torch.where(mask == True)[0]
            mask = store.edge_index
            mask = torch.isin(mask, vals).sum(axis=0).bool()
        elif store.is_edge_attr(attr):
            keys = store.node_attrs()
            vals = store.edge_index.reshape(-1).unique()
            mask = torch.zeros(store.num_nodes, dtype=bool)
            mask[vals] = True
            num_nodes = int(mask.sum())

        for key, dim in store._cat_dims(keys).items():
            store[key] = select(store[key], mask, dim)

        if "num_nodes" in store:
            store.num_nodes = num_nodes

    return out
