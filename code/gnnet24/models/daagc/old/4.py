import logging as log
import random
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset
from torch_geometric.typing import Adj, OptTensor

from ..gat import GAT
from ..kmeans import KMeans
from ...utils import ABCModel
from ...utils import EarlyStop
from ...utils.evaluate import evaluate
from ...utils.mask import get_submask
# from ...utils.split import split_dataset
from ...utils import (
     get_utg_node_index,
     get_utg_array,
     get_utg_tensor
)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log.basicConfig(format=LOG_FORMAT, level=log.INFO)


class DAAGC(ABCModel):
    """
    Implementation of **Deep Attentional Augmented with Graph Clustering**,
    based on the DAEGC model proposed by [Wang et al., 2019](https://arxiv.org/abs/1906.06532).

    This model uses a 2-layer GAT (Graph Attention Network) for message passing,
    Adam (Adaptive Moment Estimation) as optimizer, and K-means for clustering.

    On **pretrain** stage, the model is trained to predict the adjacency matrix `A`
    based on input node-level attribute features `x`, normalized adjacency matrix `adj`,
    and node transition matrix `M`. The loss is the binary cross-entropy between
    predicted adjacency matrix `A` and ground truth adjacency matrix `adj_label`.
    The K-means algorithm is used to cluster the obtained node embeddings `z`.

    On **train** stage, the model is trained to minimize the loss function, which is
    the sum of the binary cross-entropy between predicted adjacency matrix `A` and
    ground truth adjacency matrix `adj_label`, and the Kullback-Leibler divergence
    between soft cluster assignments `Q` and target distribution `p`, weighted by
    parameter `gamma`. The KL divergence considers the similarity between obtained
    node embeddings `z` and cluster centroids, and the target distribution `p` is
    calculated from soft cluster assignments `Q` every `update_interval` epochs.
    The evaluation considers the most confident hard cluster assignments `q`, i.e.,
    the cluster with the highest probability (similarity to centroid) for each node.

    Both stages learn from the training set only, with early stopping based on the
    validation set. Test is perfomed once on the test set with the best weights.

    :param dataset: PyG dataset.
    :param stage: Stage of training, either 'pretrain' or 'train'.
    :param state_dict_path: File name to store weights (required for early stopping).
        Default: 'weights_{stage}.pkl'.

    - Reference paper:
        [Attributed Graph Clustering: A Deep Attentional Embedding Approach](https://arxiv.org/abs/1906.06532) (2019)

    :param num_features: Number of input features.
    :param hidden_size: Number of hidden units.
    :param embedding_size: Dimension of the node embeddings.
    :param n_clusters: Number of clusters.
    :param lr: Learning rate.
    :param weight_decay: Weight decay (L2 regularization).
    :param alpha: Alpha for LeakyReLU (default: 0.2).
    :param gamma: Weight for clustering loss using KL divergence (default: 10).
    :param t: Number of hops for the node transition matrix (default: 2).
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
    :param split: Whether to split the dataset into train, validation, and test sets.
        If None, dataset will be split only if masks exist in the dataset (default).
    :param patience: Number of epochs to wait for improvement before stopping (default: 5).
        If None, the training loop will not stop.
    :param monitor: Metric to use for early stopping (default: 'acc').
    :param device: Device to use for computation.
        Default: 'cuda' if available; otherwise, 'cpu'.
    :param seed: Random seed number (optional).
    """
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        embedding_size: int,
        n_clusters: int,
        lr: float,
        weight_decay: float,
        max_epoch: int,
        min_epoch: Optional[int] = None,
        alpha: float = 0.2,
        beta: float = 1,
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
        self.beta = beta
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
        self.gat = GAT(num_features,
                       hidden_size,
                       embedding_size,
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

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, embedding_size))
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

    def get_Q(self, z: Tensor):
        """
        Compute soft cluster probabilities.

        :param z: Node embeddings.
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def get_M(self, adj: Adj):
        """
        Compute node transition matrix.

        :param adj: Normalized adjacency matrix.
        """
        adj_numpy = adj.cpu().numpy()
        tran_prob = normalize(adj_numpy, norm="l1", axis=0)
        M = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, self.t + 1)]) / self.t
        return torch.from_numpy(M)

    def target_distribution(self, q: Tensor) -> Tensor:
        """
        Compute target distribution `p`.

        :param q: Hard cluster assignments.
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def preprocess(self, data: Dataset) -> Dataset:
        """
        Preprocess dataset.

         By default, it moves the following attributes to the device:
            - dataset.adj: Dense normalized adjacency matrix.
            - dataset.adj_label: Ground truth adjacency matrix.
            - dataset.M: Node transition matrix.
            - dataset.x: Node-level attribute features.

        :param data: PyG dataset.
        """
        # Build adjacency matrix from edge index.
        data.adj = torch\
            .sparse_coo_tensor(data.edge_index,
                               torch.ones(data.edge_index.shape[1]),
                               torch.Size([data.x.shape[0], data.x.shape[0]]))\
            .to_dense()

        # Store original adjacency matrix.
        data.adj_label = data.adj

        # Sum adjacency matrix and identity matrix.
        data.adj += torch.eye(data.x.shape[0])

        # Normalize each row by its sum.
        data.adj = normalize(data.adj, norm="l1")

        # Convert to a tensor.
        data.adj = torch.from_numpy(data.adj).to(dtype=torch.float)

        # Store node transition matrix to calculate attention.
        data.M = self.get_M(data.adj)

    def preprocess_temporal(self, data: Dataset, temporal: bool = None):
        """
        Preprocess dataset.

        :param data: PyG dataset.
        :param temporal: Whether to consider temporal masks.
            If None, it will be inferred from the dataset.
        """
        data.val_mask = data.train_mask + data.val_mask
        data.val_submask = get_submask(data.val_mask + data.train_mask, data.val_mask)

        if temporal or (temporal is None and hasattr(data, "time")):
            # Store temporal masks.
            data.temporal_node_index_train = get_temporal_node_index(data, mask=data.train_mask)
            data.temporal_node_index_val = get_temporal_node_index(data, mask=data.val_mask)
            data.temporal_node_giindex_test = get_temporal_node_index(data, mask=data.test_mask)

            # Store temporal ground truth arrays.
            y = data.y.cpu().numpy()
            data.y_train = get_temporal_array(y[data.train_mask], data.temporal_node_index_train)
            data.y_val = get_temporal_array(y[data.val_mask], data.temporal_node_index_val)
            data.y_test = get_temporal_array(y[data.test_mask], data.temporal_node_index_test)

    def pretrainer(self):
        self.train()

        # Forward pass.
        A, z = self.gat(dataset.x[train],
                        dataset.adj[train][:, train],
                        dataset.M[train][:, train])

        # Backward pass.
        loss = F.binary_cross_entropy(A.view(-1), dataset.adj_label[train][:, train].view(-1))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Evaluation.
        z = get_temporal_tensor(z, temporal_node_index_train, how=self.agg_method)
        y_pred = self.kmeans.fit_predict(z)
        eva = evaluate(y_train, y_pred)
        losses["train"].append(eva|dict(loss=loss.item()))

        # Validation.
        self.eval()

        with torch.no_grad():
            A, z = self.gat(dataset.x[val+train],
                            dataset.adj[val+train][:, val+train],
                            dataset.M[val+train][:, val+train])

            loss = F.binary_cross_entropy(A[val_][:, val_].view(-1), dataset.adj_label[val][:, val].view(-1))
            z = get_temporal_tensor(z[val_], temporal_node_index_val, how=self.agg_method)
            y_pred = self.kmeans.fit_predict(z)
            eva = evaluate(y_val, y_pred)
            losses["val"].append(eva|dict(loss=loss.item()))

        # Early stop.
        if self.es(eva):
            self.load_state_dict(state_dict_path)
            log.info(f"Early stopping at epoch {self.es.epoch} after "
                        f"{self.es.counter} epochs of no improvement "
                        f"(best epoch: {self.es.best_epoch}).")
            # break

        elif epoch == self.es.best_epoch:
            self.save_state_dict(state_dict_path)

        # Test.
        with torch.no_grad():
            _, z = self.gat(dataset.x,    # dataset.x[test],
                            dataset.adj,  # dataset.adj[test][:, test],
                            dataset.M)    # dataset.M[test][:, test])

            z = get_temporal_tensor(z[test], temporal_node_index_test, how=self.agg_method)
            y_pred = self.kmeans.fit_predict(z)

            eva = evaluate(y_test, y_pred)
            log.info(self.es._info("test", eva))
            losses["test"] = eva

        return losses

    def trainer(self, dataset: Dataset) -> dict:
        """ Training loop. """
        train_mask = getattr(self, "train_mask", None)

        # Forward pass.
        A, z, Q = self(dataset.x[train],
                        dataset.adj[train][:, train],
                        dataset.M[train][:, train])

        Q = get_temporal_tensor(Q.detach(), temporal_node_index_train, how=self.agg_method)
        y_pred = Q.detach().data.cpu().numpy().argmax(1)

        if epoch % self.update_interval == 0:
            p = self.target_distribution(Q.detach())

        # Backward pass.
        r_loss = F.binary_cross_entropy(A.view(-1), dataset.adj_label[train][:, train].view(-1))
        c_loss = F.kl_div(Q.log(), p, reduction="batchmean")
        loss = (self.beta * r_loss) + (self.gamma * c_loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Evaluation.
        eva = evaluate(y_train, y_pred)
        return eva|dict(loss=loss.item(), r_loss=r_loss.item(), c_loss=c_loss.item())

    @torch.no_grad()
    def evaluator(self, dataset: Dataset) -> dict:
        """ Validation loop. """
        val_mask = getattr(self, "val_mask", None)

        A, z, Q = self(dataset.x[val+train_mask],
                       dataset.adj[val+train][:, val+train_mask],
                       dataset.M[val+train][:, val+train_mask])

        Q = get_temporal_tensor(Q[self.val_submask].detach(), temporal_node_index_val, how=self.agg_method)
        y_pred = Q.detach().data.cpu().numpy().argmax(1)

        if epoch % self.update_interval == 0:
            p_ = self.target_distribution(Q.detach())

        r_loss = F.binary_cross_entropy(A[self.val_submask][:, self.val_submask].view(-1), dataset.adj_label[val][:, val].view(-1))
        c_loss = F.kl_div(Q.log(), p_, reduction="batchmean")
        loss = (self.beta * r_loss) + (self.gamma * c_loss)

        eva = evaluate(y_val, y_pred)
        return eva|dict(loss=loss.item(), r_loss=r_loss.item(), c_loss=c_loss.item())

    @torch.no_grad()
    def tester(self, dataset: Dataset) -> dict:
        """ Test loop. """
        test_mask = getattr(self, "test_mask", None)

        A, z, Q = self(dataset.x,    # dataset.x[test_mask],
                       dataset.adj,  # dataset.adj[test_mask][:, test_mask],
                       dataset.M)    # dataset.M[test_mask][:, test_mask])

        Q = get_temporal_tensor(Q[dataset.test_mask].detach(),
                                self.temporal_node_index_test,
                                how=self.agg_method)

        y_pred = Q.detach().data.cpu().numpy().argmax(1)

        eva = evaluate(y_test, y_pred)
        return eva

    def pretrain(self):
        self.pretrain_ = True

    @property
    def cluster_layer(self):
        return self.kmeans.cluster_centers
