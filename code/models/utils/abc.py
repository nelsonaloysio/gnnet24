from abc import ABCMeta, abstractmethod
# from typing import Optional

import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj
# import torch


class ABCLayer(nn.Module, metaclass=ABCMeta):
    """
    Abstract layer class.
    """
    @abstractmethod
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"

    @abstractmethod
    def forward(self, x: Tensor, adj: Adj):
        ...


class ABCModel(nn.Module, metaclass=ABCMeta):
    """
    Abstract model class.
    """
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Abstract forward pass function.
        """
        ...

    '''
    def load_state_dict(self, path: str, parameters: Optional[list] = None, map_location: str = "cpu") -> None:
        """
        Load state dict from file.

        :param path: Path to the file.
        :param parameters: Parameters to load from the state dict (optional).
        :param map_location: Device to map the state dict.
        """
        checkpoint = torch.load(path, map_location=map_location)

        if parameters is None:
            parameters = self.state_dict().keys()

        for parameter in parameters:
            if parameter in checkpoint:
                self._parameters[parameter] = checkpoint[parameter]
                log.debug(f"Weights for {parameter} loaded from checkpoint.")
            else:
                log.debug(f"Weights for {parameter} not found in checkpoint.")

    def save_state_dict(self, path: str) -> None:
        """
        Save state dict to file.

        :param path: Path to the file.
        """
        torch.save(self.state_dict(), path)
    '''