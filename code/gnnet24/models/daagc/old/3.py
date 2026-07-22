import logging as log
import random
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from sklearn.preprocessing import normalize
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset
from torch_geometric.typing import Adj, OptTensor

from ..gat import GAT
from ..kmeans import KMeans
from ...utils.early_stop import EarlyStop
from ...utils.evaluate import evaluate
from ...utils.mask import get_submask
from ...utils.split import split_dataset
from ...utils.utg import (
    get_utg_node_index,
    get_utg_array,
    get_utg_tensor,
)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log.basicConfig(format=LOG_FORMAT, level=log.INFO)


class DAEGC(ABCModel):
    """
    Implementation of **Deep Attentional Encoder with Graph Clustering**.

    This model uses a 2-layer GAT (Graph Attention Network) for message passing,
    Adam (Adaptive Moment Estimation) as optimizer, and K-means for clustering.

    - Reference paper:
        [Attributed Graph Clustering: A Deep Attentional Embedding Approach](https://arxiv.org/abs/1906.06532) (2019)

    :param in_channels: Number of input features.
    :param hidden_channels: Number of hidden units.
    :param out_channels: Dimension of the node embeddings.
    :param n_clusters: Number of clusters.
    :param lr: Learning rate.
    :param weight_decay: Weight decay (L2 regularization).
    :param alpha: Alpha for LeakyReLU (default: 0.2).
    :param gamma: Weight for clustering loss using KL divergence (default: 10).
    :param v: Divisor for soft cluster probabilities (default: 1).
    :param update_interval: Interval to update hard cluster assignments (default: 5).
    :param init: Method for initialization of centroids (default: 'k-means++').
        - 'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
        - 'random': choose k observations (rows) at random from data for the initial centroids.
    :param n_init: Number of time the k-means algorithm will be run with different centroid seeds (default: 20).
        The final results will be the best output of `n_init` consecutive runs in terms of inertia.
    :param max_iter: Maximum number of iterations of the k-means algorithm for a single run (default: 300).
    :param distance: Distance function to use for K-Means.
        - 'l2': Euclidean distance (default).
        - 'l1': Manhattan distance.
        - 'cosine': Cosine similarity.
        - 'dotproduct': Dot product similarity.
    :param max_epoch: Maximum number of epochs for training and pretraining.
    :param min_epoch: Minimum number of epochs for training and pretraining (before early stop).
    :param agg_method: Aggregation method for temporal node embeddings (default: 'average').
    :param split: Whether to split the data into train, validation, and test sets.
        If None, data will be split only if masks exist in the data (default).
    :param patience: Number of epochs to wait for improvement before stopping (default: 5).
        If None, the training loop will not stop.
    :param monitor: Metric to use for early stopping (default: 'acc').
    :param device: Device to use for computation.
        Default: 'cuda' if available; otherwise, 'cpu'.
    :param seed: Random seed number (optional).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_clusters: int,
        lr: float,
        weight_decay: float,
        max_epoch: int,
        min_epoch: Optional[int] = None,
        alpha: float = 0.2,
        gamma: float = 10,
        t: int = 2,
        v: int = 1,
        update_interval: int = 5,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 20,
        max_iter: int = 300,
        distance: Optional[Literal["l2", "l1", "cosine", "dotproduct"]] = None,
        soft: Optional[bool] = True,
        agg_method: Optional[str] = "average",
        split: Optional[bool] = None,
        patience: Optional[int] = 5,
        monitor: Optional[str] = "acc",
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.t = t
        self.v = v
        self.gamma = gamma
        self.split = split
        self.max_epoch = max_epoch
        self.agg_method = agg_method
        self.update_interval = update_interval

        self.es = EarlyStop(patience, max_epoch=max_epoch, min_epoch=min_epoch, monitor=monitor)
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed number.
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Use a 2-layer GAT (Graph Attention Network) for message passing.
        self.gat = GAT(in_channels,
                       hidden_channels,
                       out_channels,
                       alpha).to(self.device)

        # Use Adam (Adaptive Moment Estimation) as optimizer.
        self.optimizer = Adam(self.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)

        # Use K-means as clustering algorithm.
        self.kmeans = KMeans(n_clusters=n_clusters,
                             init=init,
                             n_init=n_init,
                             max_iter=max_iter,
                             distance=distance,
                             soft=soft,
                             seed=self.seed,
                             device=device)

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, out_channels))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # TODO: remove lines below.
        # Load into device.
        # self.to(self.device)

    def forward(self, x: Tensor, adj: Adj, M: OptTensor = None) -> Tensor:
        """
        Forward pass. Returns predicted adjacency matrix `A`,
        node embeddings `z`, and soft cluster probabilities `Q`.

        :param x: Node-level attribute features.
        :param adj: Normalized adjacency matrix.
        :param M: Node transition matrix (optional).
        """
        A, z = self.gat(x, adj, M)
        q = self.get_Q(z.to(self.device))
        return A, z, q

    @staticmethod
    def get_A(data: Dataset) -> Dataset:
        """
        Get adjacency matrix from PyG data.

        :param data: PyG data.
        """
        num_nodes = pyg.utils.num_nodes.maybe_num_nodes(
            data.edge_index, getattr(data, "num_nodes", None))

        # Normalized sparse adjacency matrix,
        # where A[i, j] = 1 / (degree(j) + 1).
        A = 1 / (pyg.utils.degree(data.edge_index[0]) + 1)

        # Convert to dense adjacency matrix.
        A = pyg.utils.to_dense_adj(data.edge_index, edge_attr=A).squeeze()

        adj = pyg.utils.to_dense_adj(data.edge_index).

        adj = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.shape[1]),
            torch.Size([num_nodes, num_nodes])
        ).to_dense()

        # Store original adjacency matrix.
        adj_label = adj.to(dtype=torch.bool)

        # Sum with identity matrix (adding one to diagonal entries).
        adj += torch.eye(adj.shape[0])

        # Normalize each row by its sum.
        adj = normalize(adj, norm="l1")

        # Convert to tensor.
        adj = torch.from_numpy(adj).to(dtype=torch.float)

        return adj, adj_label

    def get_M(self, adj: Adj, t: int = 2) -> Tensor:
        """
        Compute node transition matrix.

        :param adj: Normalized adjacency matrix.
        :param t: Number of hops for the node transition matrix (default: 2).
        """
        adj = adj.cpu().numpy()
        tran_prob = normalize(adj, norm="l1", axis=0)
        M = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
        return torch.from_numpy(M).to(dtype=torch.float)

    def get_P(self, q: Tensor) -> Tensor:
        """
        Compute target distribution `p`.

        :param q: Hard cluster assignments.
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_Q(self, z: Tensor) -> Tensor:
        """
        Compute soft cluster probabilities.

        :param z: Node embeddings.
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def trainer(self, data: Dataset, state_dict_path: str = "weights.pkl") -> dict:
        """
        Trains model minimizing the binary cross-entropy loss between the predicted
        adjacency matrix `A` and the ground truth adjacency matrix `adj_label`, plus
        the Kullback-Leibler divergence between soft cluster assignments `Q` and the
        target distribution `p`, weighted by parameter `gamma`. KL here considers
        the similarity between obtained node embeddings `z` and cluster centroids,
        and the target distribution `p` is calculated from soft cluster assignments
        `Q` at every `update_interval` epochs.

        :param data: PyG data.
        :param state_dict_path: File name to store weights (required for early stopping).
            Default: 'weights_{stage}.pkl'.
        """
        self.es.reset()
        eva = {"train": [], "val": [], "test": None}
        train, val, test = None, None, None

        with torch.no_grad():
            _, z = self.gat(data.x[train],
                            data.adj[train][:, train],
                            data.M[train][:, train])

            # z = get_utg_tensor(z, utg_node_index_train, how=self.agg_method)
            y_pred = self.kmeans.fit(z).predict(z)
            eva = evaluate(y[train], y_pred)
            eva["pretrain"] = eva

        self.cluster_layer.data = self.kmeans.cluster_centers

        for epoch in range(self.max_epoch):
            self.train()

            # Forward pass.
            A, z, Q = self(data.x[train],
                           data.adj[train][:, train],
                           data.M[train][:, train])

            # Q = get_utg_tensor(Q.detach(), utg_node_index_train, how=self.agg_method)
            y_pred = Q.detach().data.cpu().numpy().argmax(1)

            if epoch % self.update_interval == 0:
                p = self.get_P(Q.detach())

            # Backward pass.
            r_loss = F.binary_cross_entropy(A.view(-1), data.adj_label[train][:, train].view(-1))
            c_loss = F.kl_div(Q.log(), p, reduction="batchmean")
            loss = r_loss + (self.gamma * c_loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            eva = evaluate(y[train], y_pred)
            eva["train"].append(eva | dict(loss=loss.item(), r_loss=r_loss.item(), c_loss=c_loss.item()))

            # Validation.
            self.eval()

            with torch.no_grad():
                A, z, Q = self(data.x[val],
                               data.adj[val][:, val],
                               data.M[val][:, val])

                # Q = get_utg_tensor(Q[val_].detach(), utg_node_index_val, how=self.agg_method)
                y_pred = Q.detach().data.cpu().numpy().argmax(1)

                if epoch % self.update_interval == 0:
                    p_ = self.get_P(Q.detach())

                r_loss = F.binary_cross_entropy(A[val][:, val].view(-1), data.adj_label[val][:, val].view(-1))
                c_loss = F.kl_div(Q.log(), p_, reduction="batchmean")
                loss = r_loss + (self.gamma * c_loss)

                eva = evaluate(y[val], y_pred)
                eva["val"].append(eva | dict(loss=loss.item(), r_loss=r_loss.item(), c_loss=c_loss.item()))

            # Early stop.
            if self.es(eva):
                self.load_state_dict(state_dict_path)
                break
            elif epoch == self.es.best_epoch:
                self.save_state_dict(state_dict_path)

        # Test.
        with torch.no_grad():
            _, z, Q = self(data.x,    # data.x[test],
                           data.adj,  # data.adj[test][:, test],
                           data.M)    # data.M[test][:, test])

            # Q = get_utg_tensor(Q[test].detach(), utg_node_index_test, how=self.agg_method)
            y_pred = Q.detach().data.cpu().numpy().argmax(1)

            eva = evaluate(y[test], y_pred)
            log.info(self.es._info("test", eva))
            eva["test"] = eva

        return eva

    def pretrainer(self, data: Dataset, state_dict_path: str = "weights.pkl") -> dict:
        """
        Trains model minimizing the binary cross-entropy loss between the predicted
        adjacency matrix `A` and the ground truth adjacency matrix `adj_label`

        :param data: PyG data.
        :param state_dict_path: File name to store weights (required for early stopping).
            Default: 'weights_{stage}.pkl'.
        """
        evas = {"train": [], "val": [], "test": None}
        train, val, test = None, None, None
        y = data.y.cpu().numpy()

        self.es.reset()
        for epoch in range(self.max_epoch):
            self.train()

            A, z = self.gat(data.x[train], data.adj[train][:, train], data.M[train][:, train])
            loss = F.binary_cross_entropy(A.view(-1), data.adj_label[train][:, train].view(-1))
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            y_pred = self.kmeans.fit_predict(z)
            eva = evaluate(y[train], y_pred)
            evas["train"].append(eva | dict(loss=loss.item()))

            # Validation.
            self.eval()

            with torch.no_grad():
                A, z = self.gat(data.x[val], data.adj[val][:, val], data.M[val][:, val])
                loss = F.binary_cross_entropy(A[val][:, val].view(-1), data.adj_label[val][:, val].view(-1))

                y_pred = self.kmeans.fit_predict(z)
                eva = evaluate(y[val], y_pred)
                evas["val"].append(eva | dict(loss=loss.item()))

            # Early stop.
            if self.es(eva):
                self.load_state_dict(state_dict_path)
                log.info(f"Early stopping at epoch {self.es.epoch} after "
                         f"{self.es.counter} epochs of no improvement "
                         f"(best epoch: {self.es.best_epoch}).")
                break
            elif epoch == self.es.best_epoch:
                self.save_state_dict(state_dict_path)

        # Test.
        with torch.no_grad():
            _, z = self.gat(data.x[test], data.adj[test][:, test], data.M[test][:, test])
            y_pred = self.kmeans.fit_predict(z)

            eva = evaluate(y[test], y_pred)
            evas["test"] = eva

            log.info(self.es._info("test", eva))

        return evas

    @property
    def cluster_layer(self):
        return self.kmeans.cluster_centers
