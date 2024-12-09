import logging as log
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np
import scipy
import tensorflow.compat.v2 as tf
import torch

from .dmon import DMoN
from .gcn import GCN
from .utils import normalize_graph
from ...utils.convert import to_dmon
from ...utils.early_stop import EarlyStop
from ...utils.evaluate import evaluate

tf.compat.v1.enable_v2_behavior()


def main(args: Namespace) -> None:
    """ Main function. """
    # Process data (convert node features to dense, normalize the
    # graph, convert it to Tensorflow sparse tensor.
    loader = to_dmon(args.data)

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

    features = features.todense()
    n_nodes = adjacency.shape[0]
    feature_size = features.shape[1]
    graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
    graph_normalized = convert_scipy_sparse_to_sparse_tensor(
        normalize_graph(adjacency.copy()))

    # Create model input placeholders of appropriate size
    input_features = tf.keras.layers.Input(shape=(feature_size,))
    input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
    input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)
    n_clusters = len(set(labels))

    output = input_features
    for n_channels in args.architecture:
        output = GCN(n_channels)([output, input_graph])
    pool, pool_assignment = DMoN(
        n_clusters,
        collapse_regularization=args.collapse_regularization,
        dropout_rate=args.dropout_rate)([output, input_adjacency])
    model = tf.keras.Model(
        inputs=[input_features, input_graph, input_adjacency],
        outputs=[pool, pool_assignment])

    # Computes the gradients wrt. the sum of losses, returns a list of them.
    def grad(model, inputs):
        with tf.GradientTape() as tape:
            _ = model(inputs, training=True)
            loss_value = sum(model.losses)
        return model.losses, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    model.compile(optimizer, None)

    es = EarlyStop(patience=args.patience, metric="acc")

    for epoch in range(args.epochs):
        loss_values, grads = grad(model, [features, graph_normalized, graph])
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses = " ".join([f"{loss_value.numpy():.4f}" for loss_value in loss_values])

        # Obtain cluster assignments.
        _, assignments = model([features, graph_normalized, graph], training=False)
        assignments = assignments.numpy()
        y_pred = assignments.argmax(axis=1)

        es(evaluate(labels, y_pred, adj=adjacency), losses=losses)
        # if epoch == es.best_epoch:
        #     ...
        if es.early_stop():
            break

    log.info(es)

    # print("Conductance:", metrics.conductance(adjacency, clusters))
    # print("Modularity:", metrics.modularity(adjacency, clusters))
    # print("NMI:", sklearn.metrics.normalized_mutual_info_score(
    #         labels, clusters[label_indices], average_method="arithmetic"))
    # precision = metrics.pairwise_precision(labels, clusters[label_indices])
    # recall = metrics.pairwise_recall(labels, clusters[label_indices])
    # print("F1:", 2 * precision * recall / (precision + recall))


def convert_scipy_sparse_to_sparse_tensor(
    matrix):
  """Converts a sparse matrix and converts it to Tensorflow SparseTensor.

  Args:
    matrix: A scipy sparse matrix.

  Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
  """
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)


def getargs(argv: list = sys.argv[1:]) -> Tuple[Namespace, list]:
    """ Get arguments. """
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILE_PATH",
                        type=torch.load)

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

    parser.add_argument("--architecture",
                        help="Network architecture in the format `a,b,c,d`.",
                        default=[64],
                        type=lambda x: list(map(int, x.split(","))))

    parser.add_argument("--collapse-regularization",
                        help="Regularization for collapsing the graph.",
                        default=0.1,
                        type=float)

    parser.add_argument("--dropout-rate",
                        help="Dropout rate for GNN representations.",
                        default=0,
                        type=float)

    parser.add_argument("--unpooling",
                        help="Whether to unpool the graph.",
                        action="store_true")

    # parser.add_argument("--device",
    #                     help="Device to use ("cpu" or "cuda"). "
    #                          "Default: "cuda", if available.",
    #                     default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(getargs()))
