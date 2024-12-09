import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class GNNetDataset(InMemoryDataset):
    """
    Return PyG dataset object.

    :param root: Root directory.
    :param name: Dataset name.
    :param transform: Transform to apply to the data.
    :param pre_transform: Transform to apply to the data before saving.
    :param force_reload: Force reload of the dataset.
    """
    def __init__(
        self,
        root: str,
        name: str = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.name = name or self.__class__.__name__.lower()

        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

        data = self.get(0)
        self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        names = ["edge_index", "x", "y", "time", "split"]
        return names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self) -> None:
        data = Data()
        assert osp.isdir(self.raw_dir), f"Directory '{self.raw_dir}' not found."

        for name in self.raw_file_names:
            path = osp.join(self.raw_dir, name)

            if osp.exists(f"{path}.npy"):
                attr = np.load(f"{path}.npy")
                setattr(data, name, torch.from_numpy(attr))

            elif osp.exists(f"{path}.npz"):
                attrs = np.load(f"{path}.npz")

                if list(attrs.keys()) == ["arr_0"]:
                    setattr(data, name, torch.from_numpy(attrs["arr_0"]))
                else:
                    for key in attrs:
                        setattr(data, key, torch.from_numpy(attrs[key]))

        assert all(attr in data for attr in ("edge_index", "y"))
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"
