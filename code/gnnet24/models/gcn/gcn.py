import torch
import torch.nn as nn
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor

from ...utils.abc import ABCLayer, ABCModel


class GCN(ABCModel):
    """
    Implementation of a 2-layer Graph Convolutional Network.

    This GNN is used for semi-supervised node classification,
    using cross-entropy loss and Adam optimizer for training.

    Reference paper:
        - [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (2016)

    :param in_features: Number of input features.
    :param hidden_size: Number of hidden units.
    :param out_features: Number of embedding units.
    :param num_classes: Number of classes.
    :param lr: Learning rate for optimizer.
    """
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        num_classes: int,
        lr: float,
    ) -> None:
        super().__init__()

        # Convolutional layers.
        self.conv1 = pyg.nn.GCNConv(in_features, hidden_size)
        self.conv2 = pyg.nn.GCNConv(hidden_size, out_features)

        # Classifier layer.
        self.classifier = nn.Linear(out_features, num_classes)

        # Loss function.
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer function.
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> list:
        """
        Forward pass.

        :param x: Input features.
        :param edge_index: Graph edge indices.
        """
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h

    def trainer(self, data: Tensor) -> list:
        """
        Train the model.

        :param data: Input data.
        """
        # Perform forward pass.
        out, h = self.forward(data.x[data.train_mask], data.edge_index) # <-- what???

        # Calculate loss for optimizer.
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # Update based on calculated loss.
        self.optimizer.step()
        self.optimizer.zero_grad()

        return out, h, loss


class GCNLayer(ABCLayer):
    """
    Implementation of a single Graph Convolutional Network layer.

    Reference paper:
        - [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

    :param in_features: Number of input features.
    :param out_features: Number of output features.
    :param bias: If set to False, the layer will not learn an additive bias.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        self.lin = nn.Linear(
            in_features,
            out_features,
            bias=False,
            weight_initializer='glorot'
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """
        Defines forward pass for the graph neural network.

        :param x: Input features.
        :param edge_index: Graph edge indices.
        :param edge_weight: Edge weight vector.
        """
        x = self.lin(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        """
        Message passing function.

        :param x_j: Input features of neighbors.
        :param edge_weight: Edge weight vector.
        """
        if edge_weight:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        """
        Message passing and aggregation function.

        :param adj_t: Transposed adjacency matrix.
        :param x: Input features.
        """
        return pyg.utils.spmm(adj_t, x, reduce=self.aggr)
