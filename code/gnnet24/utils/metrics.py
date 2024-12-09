import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.sparse import spmatrix


def modularity(adjacency: spmatrix, clusters: list):
    """
    Computes graph modularity.

    The modularity of a graph is a measure of the structure of the network
    compared to the structure expected in a random graph. It is defined
    as the fraction of edges that fall within the given clusters minus
    the expected fraction if edges were distributed at random.

    Implementation from:
    https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/metrics.py

    See also:
    https://en.wikipedia.org/wiki/Modularity_(networks)

    :param adjacency: Input graph in terms of its sparse adjacency matrix.
    :param clusters: An (n,) int cluster vector.
    """
    degrees = adjacency.sum(axis=0).A1
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
    return result / n_edges


def conductance(adjacency: spmatrix, clusters: list):
    """
    Computes graph conductance as in Yang & Leskovec (2012).
    Returns the average conductance value of the graph clusters.

    Implementation from:
    https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/metrics.py

    :param adjacency: Input graph in terms of its sparse adjacency matrix.
    :param clusters: An (n,) int cluster vector.
    """
    inter = 0  # Number of inter-cluster edges.
    intra = 0  # Number of intra-cluster edges.
    cluster_indices = np.zeros(adjacency.shape[0], dtype=bool)
    for cluster_id in np.unique(clusters):
        cluster_indices[:] = 0
        cluster_indices[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_indices, :]
        inter += np.sum(adj_submatrix[:, cluster_indices])
        intra += np.sum(adj_submatrix[:, ~cluster_indices])
    return intra / (inter + intra)


def p4(y_true: list, y_pred: list):
    """
    Calculate the P4 metric.

    Reference paper:
        [Extending F1 metric, probabilistic approach](https://arxiv.org/abs/2210.11997)

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tn = cm.sum() - tp - fp - fn

    ## True positive rate (recall/sensitivity/hit rate)
    # tpr = tp/(tp+fn)
    ## Positive predictive value (precision)
    # ppv = tp/(tp+fp)
    ## True negative rate (specificity)
    # tnr = tn/(tn+fp)
    ## Negative predictive rate
    # npv = tn/(tn+fn)
    ## False positive rate (fallout)
    # fpr = fp/(fp+tn)
    ## False negative rate
    # fnr = fn/(tp+fn)
    ## False discovery rate
    # fdr = fp/(tp+fp)

    p4 = (4 * tp * tn) / ((4 * tp * tn) + (tp + tn) * (fp + fn))
    return p4.mean()