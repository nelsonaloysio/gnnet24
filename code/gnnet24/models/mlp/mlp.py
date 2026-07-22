from typing import Optional
import random

import numpy as np
import torch
from torch.nn import Linear, Dropout, Tanh, ReLU

ACTIVATION = {
    "tanh": Tanh,
    "relu": ReLU
}

class MLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: Optional[str] = "relu",
        dropout: Optional[float] = 0.5,
        seed: Optional[int] = None
    ) -> None:
        super(MLP, self).__init__()
        self.embed = Linear(in_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

        if activation is not None:
            self.activation = ACTIVATION[activation]()

        if dropout is not None:
            self.dropout = Dropout(0.5)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def forward(self, data):
        h = self.embed(data.x)

        if hasattr(self, "activation"):
            h = self.activation(h)

        if hasattr(self, "dropout"):
            h = self.dropout(h)

        y = self.classifier(h)
        return y, h
