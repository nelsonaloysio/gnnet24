import logging as log

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
# from sklearn.cluster import KMeans

from ..kmeans import KMeans
from ...utils.early_stop import EarlyStop
from ...utils.evaluate import evaluate


def train(
    model: nn.Module,
    data: Data,
    lr: float,
    weight_decay: float,
    gamma: float,
    update_interval: int,
    n_init: int,
    epochs: int,
    patience: int,
    seed: int,
    weights: str,
    device: str = "cpu",
):
    """
    Train DAEGC model.

    In this stage, we optimize the model parameters and the cluster centers
    jointly. We use the target probability distribution to update the cluster
    centers, and the model parameters are updated by minimizing the loss
    function, which consists of the reconstruction loss and the clustering loss.

    Note that the `device` argument applies to K-Means only (default: 'cpu').
    """
    n_clusters = len(np.unique(data.y))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    es = EarlyStop(patience=patience, metric="acc")

    train_mask, val_mask, test_mask = (
        data.train_mask, data.val_mask, data.test_mask,
    )
    y_eval = data.y[val_mask if val_mask.any() else train_mask]

    # Obtain initial cluster centers.
    with torch.no_grad():
        adj_pred, z = model(data.x[train_mask],
                            data.adj_norm[:, train_mask][train_mask],
                            data.adj_prox[:, train_mask][train_mask])

        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, seed=seed, device=device)
        kmeans.fit(z.cpu().numpy())
        model.centroids.data = kmeans.cluster_centers_.to(device)

        y_pred = kmeans.labels_
        eva = evaluate(data.y[train_mask], y_pred)
        log.info("Pretrain: %s", eva)

    for epoch in range(epochs):
        model.train()

        # Forward pass.
        adj_pred, z = model(data.x[train_mask],
                            data.adj_norm[:, train_mask][train_mask],
                            data.adj_prox[:, train_mask][train_mask])

        # Get target probability distribution.
        q = model.get_q(z)
        if epoch % update_interval == 0:
            p = q.detach()
            p = p**2 / p.sum(0)
            p = (p.t() / p.sum(1)).t()

        # Backward pass.
        r_loss = F.binary_cross_entropy(adj_pred.view(-1).cpu(),
                                        data.adj[:, train_mask][train_mask].view(-1))

        c_loss = F.kl_div(q.log(), p, reduction="batchmean")
        loss = r_loss + (gamma * c_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation.
        with torch.no_grad():
            if val_mask.any():
                adj_pred, z = model(data.x[val_mask],
                                    data.adj_norm[:, val_mask][val_mask],
                                    data.adj_prox[:, val_mask][val_mask])
                q = model.get_q(z)

            y_pred = q.detach().data.cpu().numpy().argmax(1)

            es(evaluate(y_eval, y_pred),
               r_loss=r_loss.item(),
               c_loss=c_loss.item())

        if epoch == es.best_epoch:
            torch.save(model.state_dict(), weights)
        if es.early_stop():
            break

    model.load_state_dict(torch.load(weights, map_location="cpu"))
    log.info("Best on train: %s", es.best)

    # Test.
    if test_mask.any():
        with torch.no_grad():
            adj_pred, z = model(data.x[test_mask],
                                data.adj_norm[:, test_mask][test_mask],
                                data.adj_prox[:, test_mask][test_mask])

        q = model.get_q(z)
        y_pred = q.detach().data.cpu().numpy().argmax(1)
        eva = evaluate(data.y[test_mask], y_pred)
        log.info("Test on train: %s", eva)
