import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric as pyg
from sklearn.linear_model import LogisticRegression
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER


class Node2Vec(torch.nn.Module):
    """
    Generates embeddings from a graph using the Node2Vec algorithm.

    This implementation is CPU or GPU-bound and relies on PyTorch.

    For more information, please check the
    [documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html).

    :param data: Graph to generate embeddings from.
    :param embedding_dim: Dimension of the embeddings.
    :param walk_length: Length of each walk.
    :param context_size: The actual context window size considered for
        positive samples. This parameter increases the effective sampling
        rate by reusing samples across different source nodes.
    :param walks_per_node: Number of walks per node.
    :param num_negative_samples: Number of negative samples to use.
    :param p: Return probability parameter. Lower values promote BFS walks,
        sampling nodes at the same distance as the previous node. Default: 1.0.
    :param q: Exploration probability parameter. Lower values promote walks,
        sampling nodes further away from the previous node. Default: 1.0.
    :param lr: Learning rate for the optimizer. Default: 0.01.
    :param batch_size: Number of nodes to sample in a single batch.
    :param num_workers: Number of parallel workers. Default: 1.
    :param sparse: Whether to use sparse gradients. Default: False.
    :param seed: Random seed number for predictable randomness.
    """
    def __init__(
        self,
        data: Dataset,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int,
        num_negative_samples: int,
        p: float = 1.0,
        q: float = 1.0,
        lr: float = 0.01,
        batch_size: int = 128,
        num_workers: Optional[int] = 1,
        sparse: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

        if WITH_PYG_LIB and p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        elif WITH_TORCH_CLUSTER:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            if p == 1.0 and q == 1.0:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires either the 'pyg-lib' or "
                                  f"'torch-cluster' package")
            else:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires the 'torch-cluster' package")

        self.num_nodes = pyg.utils.num_nodes.maybe_num_nodes(data.edge_index, getattr(data, "num_nodes", None))
        row, col = pyg.utils.sort_edge_index(data.edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = pyg.utils.sparse.index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.embedding = torch.nn.Embedding(
            self.num_nodes,
            embedding_dim,
            sparse=sparse
        )

        self.reset_parameters()

        self.loader = torch.utils.data.DataLoader(
            range(self.num_nodes),
            collate_fn=self.sample,
            batch_size=batch_size,
            num_workers=num_workers or 1,
            shuffle=True,
        )

        self.optimizer = getattr(torch.optim, "SparseAdam" if sparse else "Adam")(
            list(self.parameters()),
            lr=lr,
        )

        self.to(self.device)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        r"""Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch,
                                 self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    @torch.no_grad()
    def test(
        self,
        y_train: Tensor,
        z_train: Tensor,
        y_test: Tensor,
        z_test: Tensor,
        solver: str = "lbfgs",
        multi_class: str = "auto",
        *args,
        **kwargs,
    ) -> float:
        r"""Evaluates latent space quality via a logistic regression downstream
        task.

        See [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
        """
        clf = LogisticRegression(
            solver=solver,
            multi_class=multi_class,
            *args,
            **kwargs
        )
        clf.fit(z_train, y_train)
        return clf.score(z_test, y_test)
