import logging as log
import math
import os
import pickle as pkl
import random
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn as nn
from networkx.generators.degree_seq import expected_degree_graph
from scipy import sparse
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm.auto import tqdm

from . import utils
from .Lossfunction import loss_modularity_trace
from .UCODEncoder import GCN
from ..dmon.utils import normalize_graph, convert_scipy_sparse_to_sparse_tensor
from ...utils.convert import to_dmon
from ...utils.early_stop import EarlyStop
from ...utils.evaluate import evaluate

METRICS = ["acc", "ami", "ari", "f1", "p4", "nmi", "modularity", "conductance"]
# METRICS = ["ami", "ari", "f1", "p4", "nmi", "modularity", "conductance"]
PATH = os.path.dirname(os.path.abspath(__file__))


def main(args: Namespace) -> None:
    """ Main function. """
    loader = to_dmon(args.data)

    # features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj_metric, m = datapreprocessing('kipf', PATH, args.name, 1)
    # return features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj_metric, m

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    adjacency = scipy.sparse.csr_matrix(
        (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
        shape=loader["adj_shape"]
    )
    features = scipy.sparse.csr_matrix(
        (loader["feature_data"], loader["feature_indices"], loader["feature_indptr"]),
        shape=loader["feature_shape"]
    )
    label_indices = loader["label_indices"]
    labels = loader["labels"]

    assert adjacency.shape[0] == features.shape[0],\
        "Adjacency and feature size must be equal!"
    assert labels.shape[0] == label_indices.shape[0],\
        "Labels and label_indices size must be equal!"

    weights_file = f"{PATH}/dataset/Weights-{args.name}.pt"
    B_file = f"{PATH}/dataset/Modularity-{args.name}.npy"
    if os.path.isfile(B_file):
        B = np.load(B_file)
    else:
        network = nx.from_scipy_sparse_matrix(adjacency)
        B = utils.get_B(network)
        np.save(B_file, B)

    features, adj, B, true_labels = utils.convert_torch_npz(adjacency, features, B, labels)
    # label_mask = np.ones_like(labels)
    label_mask = np.ones((labels.shape[0], labels.shape[0]))
    nb_nodes = labels.shape[0]
    ft_size = features.shape[-1]
    adj_metric = adjacency
    m = loader["adj_data"].shape[0]//2

    # NOTE: DMoN uses sparse adjacency matrix in tuple format.
    # n_nodes = adjacency.shape[0]
    # feature_size = features.shape[1]
    # graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    # graph_normalized = convert_scipy_sparse_to_sparse_tensor(
    #     normalize_graph(adjacency.copy()))

    # data_name = args.name
    n_communities = len(set(true_labels))
    hid_units = args.hid_units or n_communities
    hid_dimensions = args.hid_dimension
    sigmoidlogit = nn.Sigmoid()

    model = GCN(ft_size, hid_units, nb_nodes, hid_dimensions)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        B = B.cuda()
        adj = adj.cuda()

    es = EarlyStop(patience=args.patience, metric="modularity")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(features, adj)
        loss = loss_modularity_trace(logits, B, n_communities, hid_units, m, 'non-overlap')

        loss.backward()
        optimizer.step()
        model.eval()

        logits = model(features, adj)
        logits = sigmoidlogit(logits)
        y_pred = torch.argmax(logits[0], dim=1).detach().cpu().numpy()

        eva = evaluate(labels, y_pred, adj=adjacency, metrics=METRICS)
        es(eva, loss=loss)

        if epoch == es.best_epoch:
            torch.save(model.state_dict(), weights_file)
        if es.early_stop():
            break

    log.info("best: %s", es.best)

    # # Baseline comparison against K-Means clustering on the learned embeddings.
    # cpu_logits = logits.view(nb_nodes, hid_units).detach().cpu().numpy()
    # kmeans_model = KMeans(
    #     init='k-means++',
    #     n_clusters=n_communities,
    #     random_state=0
    # ).fit(cpu_logits)
    # kmeans_pred = kmeans_model.labels_
    # kmeans_eva = evaluate(labels, kmeans_pred, adj=adjacency, metrics=METRICS)
    # log.info("kmeans: %s", kmeans_eva)


def getargs(argv: list = sys.argv[1:]) -> Tuple[Namespace, list]:
    """ Get arguments. """
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILE_PATH",
                        type=torch.load)

    parser.add_argument("--name",
                        help="Dataset name to save modularity matrix.")

    parser.add_argument("--epochs",
                        help="Maximum number of epochs. Default: 1000.",
                        type=int,
                        default=1000)

    parser.add_argument("--patience",
                        help="Epochs to wait for improvement. Optional.",
                        type=int,
                        default=None)

    parser.add_argument("--lr",
                        dest="learning_rate",
                        help="Learning rate of the optimizer. Default: 1e-3.",
                        default=1e-3,
                        type=float)

    parser.add_argument("--hidden-dim",
                        dest="hid_dimension",
                        help="Number of hidden dimensions. Default: 500.",
                        default=500,
                        type=int)

    parser.add_argument("--hidden-units",
                        default=None,
                        dest="hid_units",
                        help="Number of hidden units. If unset, defaults to number of clusters.",
                        type=int)

    parser.add_argument("--weight-decay",
                        help="Weight decay (L2 penalty) of the optimizer. Default: 0.1.",
                        default=0.1,
                        type=float)

    parser.add_argument("--seed",
                        help="Random seed number.",
                        type=int)

    parser.add_argument("--device",
                        help="Device to use ('cpu' or 'cuda'). "
                             "Default: 'cuda', if available.",
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args(argv)


def datapreprocessing(data_type, path, dataset, exist_B):
    if data_type == 'kipf':
        adj, features, labels = utils.load_data(dataset)
        with open(path + "/dataset/ind." + str(dataset) + ".graph", 'rb') as f:
            Graph = pkl.load(f, encoding='latin1')
        network = nx.Graph(Graph)
        label_mask = np.ones_like(labels)
    elif data_type == "npz":
        adj, features, labels, label_mask = utils.load_npz_to_sparse_graph(path + '/dataset/' + dataset + '.npz')
        network = nx.from_scipy_sparse_matrix(adj)
        features = sparse.csr_matrix(features)
    adj_metric=adj
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    m = len(network.edges)

    if exist_B == 0:
        B=utils.get_B(network)
        np.save(path+'/dataset/Modularity-'+dataset, B)
    else:
        B = np.load(path+'/dataset/Modularity-'+dataset+'.npy')

    if data_type=="kipf":
        features, adj, B, true_labels = utils.convert_torch_kipf(adj, features, B, labels)
    elif data_type=="npz":
        features, adj, B, true_labels = utils.convert_torch_npz(adj, features, B, labels)

    return features, adj, B, true_labels, label_mask, nb_nodes, ft_size, adj_metric, m


if __name__ == "__main__":
    sys.exit(main(getargs()))
