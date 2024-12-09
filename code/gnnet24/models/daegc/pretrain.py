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


def pretrain(
    model: nn.Module,
    data: Data,
    lr: float,
    weight_decay: float,
    n_init: int,
    epochs: int,
    patience: int,
    seed: int,
    weights: str,
    device: str = "cpu",
) -> None:
    """
    Pretrain DAEGC model.

    In this stage, we optimize the model parameters by minimizing the loss
    function, which consists of the reconstruction loss only. The cluster
    labels are obtained by K-Means clustering on the learned node embeddings.

    Note that the `device` argument applies to K-Means only (default: 'cpu').
    """
    n_clusters = len(np.unique(data.y))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    es = EarlyStop(patience=patience, metric="acc")

    train_mask, val_mask, test_mask = (
        data.train_mask, data.val_mask, data.test_mask,
    )
    y_eval = data.y[val_mask if val_mask.any() else train_mask]

    for epoch in range(epochs):
        model.train()

        # Forward pass.
        adj_pred, z = model(data.x[train_mask],
                            data.adj_norm[:, train_mask][train_mask],
                            data.adj_prox[:, train_mask][train_mask])

        # Compute loss.
        loss = F.binary_cross_entropy(adj_pred.view(-1).cpu(),
                                      data.adj[:, train_mask][train_mask].view(-1))

        # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation.
        with torch.no_grad():
            if val_mask.any():
                adj_pred, z = model(data.x[val_mask],
                                    data.adj_norm[:, val_mask][val_mask],
                                    data.adj_prox[:, val_mask][val_mask])

            y_pred = KMeans(n_clusters=n_clusters,
                            n_init=n_init,
                            seed=seed,
                            device=device).fit_predict(z)

            es(evaluate(y_eval, y_pred), loss=loss.item())

        if epoch == es.best_epoch:
            torch.save(model.state_dict(), weights)
        if es.early_stop():
            break

    model.load_state_dict(torch.load(weights, map_location="cpu"))
    log.info("Best on pretrain: %s", es.best)

    # Test.
    if test_mask.any():
        with torch.no_grad():
            adj_pred, z = model(data.x[test_mask],
                                data.adj_norm[:, test_mask][test_mask],
                                data.adj_prox[:, test_mask][test_mask])

        y_pred = KMeans(n_clusters=n_clusters,
                        n_init=n_init,
                        seed=seed,
                        device=device).fit_predict(z)

        eva = evaluate(data.y[test_mask], y_pred)
        log.info("Test on pretrain: %s", eva)
