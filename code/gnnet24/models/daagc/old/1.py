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


class DAAGC(ABCModel):
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

    def data_process(self, data: Dataset) -> Dataset:
        """
        Preprocess dataset.

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
        # Sum with identity matrix (adding one to diagonal entries).
        data.adj += torch.eye(data.x.shape[0])
        # Normalize each row by its sum.
        data.adj = normalize(data.adj, norm="l1")
        # Convert to a tensor.
        data.adj = torch.from_numpy(data.adj).to(dtype=torch.float)
        # Store node transition matrix to calculate attention.
        data.M = self.get_M(data.adj)
        return data

    def target_distribution(self, q: Tensor) -> Tensor:
        """
        Compute target distribution `p`.

        :param q: Hard cluster assignments.
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def trainer(self, dataset: Dataset, stage: Literal["train", "pretrain"], state_dict_path: Optional[str] = None) -> dict:
        """
        Function to train or pretrain the model.

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
        """
        assert stage in ("pretrain", "train"),\
               "Invalid argument for `stage`, expects either 'pretrain' or 'train'."

        if state_dict_path is None:
            state_dict_path = f"weights_{stage}.pkl"

        assert state_dict_path.endswith(".pkl"),\
               "Invalid file name for `state_dict_path`, expects a '.pkl' file."

        self.es.reset()
        losses = {"train": [], "val": [], "test": None}
        train, val, test = split_dataset(dataset, self.split)
        val_ = get_submask(val+train, val)

        temporal_node_index_train = get_temporal_node_index(dataset, mask=train)
        temporal_node_index_val = get_temporal_node_index(dataset, mask=val)
        temporal_node_index_test = get_temporal_node_index(dataset, mask=test)

        y = dataset.y.cpu().numpy()
        y_train = get_temporal_array(y[train], temporal_node_index_train)
        y_val = get_temporal_array(y[val], temporal_node_index_val)
        y_test = get_temporal_array(y[test], temporal_node_index_test)

        log.info(f"Dataset{f' {dataset.name}' if hasattr(dataset, 'name') else ''}: "\
                 f"V={dataset.x.shape[0]}, "\
                 f"X={dataset.x.shape[1]}, "\
                 f"E={dataset.edge_index[0].shape[0]} (" +\
                 f"train={train.sum()/dataset.x.shape[0]:.2f}, "
                 f"val={val.sum()/dataset.x.shape[0]:.2f}, "
                 f"test={test.sum()/dataset.x.shape[0]:.2f})...")

        def _pretrain():
            for epoch in range(self.max_epoch):
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
                    break
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
                log.info(self.es.info("test", eva))
                losses["test"] = eva

            return losses

        def _train():
            with torch.no_grad():
                _, z = self.gat(dataset.x[train],
                                dataset.adj[train][:, train],
                                dataset.M[train][:, train])

                z = get_temporal_tensor(z, temporal_node_index_train, how=self.agg_method)
                y_pred = self.kmeans.fit(z).predict(z)
                eva = evaluate(y_train, y_pred)
                losses["pretrain"] = eva

            self.cluster_layer.data = self.kmeans.cluster_centers

            for epoch in range(self.max_epoch):
                self.train()

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
                losses["train"].append(eva|dict(loss=loss.item(), r_loss=r_loss.item(), c_loss=c_loss.item()))

                # Validation.
                self.eval()

                with torch.no_grad():
                    A, z, Q = self(dataset.x[val+train],
                                   dataset.adj[val+train][:, val+train],
                                   dataset.M[val+train][:, val+train])

                    Q = get_temporal_tensor(Q[val_].detach(), temporal_node_index_val, how=self.agg_method)
                    y_pred = Q.detach().data.cpu().numpy().argmax(1)

                    if epoch % self.update_interval == 0:
                        p_ = self.target_distribution(Q.detach())

                    r_loss = F.binary_cross_entropy(A[val_][:, val_].view(-1), dataset.adj_label[val][:, val].view(-1))
                    c_loss = F.kl_div(Q.log(), p_, reduction="batchmean")
                    loss = (self.beta * r_loss) + (self.gamma * c_loss)

                    eva = evaluate(y_val, y_pred)
                    losses["val"].append(eva|dict(loss=loss.item(), r_loss=r_loss.item(), c_loss=c_loss.item()))

                # Early stop.
                if self.es(eva):
                    self.load_state_dict(state_dict_path)
                    break
                elif epoch == self.es.best_epoch:
                    self.save_state_dict(state_dict_path)

            # Test.
            with torch.no_grad():
                _, z, Q = self(dataset.x,    # dataset.x[test],
                               dataset.adj,  # dataset.adj[test][:, test],
                               dataset.M)    # dataset.M[test][:, test])

                Q = get_temporal_tensor(Q[test].detach(), temporal_node_index_test, how=self.agg_method)
                y_pred = Q.detach().data.cpu().numpy().argmax(1)

                eva = evaluate(y_test, y_pred)
                log.info(self.es.info("test", eva))
                losses["test"] = eva

            return losses

        log.info(f"{stage.capitalize()}ing model.")
        return _pretrain() if stage == "pretrain" else _train()

    @property
    def cluster_layer(self):
        return self.kmeans.cluster_centers
