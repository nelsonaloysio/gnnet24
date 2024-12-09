from typing import Optional

from scipy.sparse import spmatrix
from sklearn.metrics import (
    accuracy_score as acc,               # ACC = (TP + TN) / (TP + TN + FP + FN)
    adjusted_mutual_info_score as ami,   # AMI = (MI - Expected_MI) / (max(H(U), H(V)) - Expected_MI)
    adjusted_rand_score as ari,          # ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    calinski_harabasz_score as chi,      # CHI = (B / W) * ((n - k) / (k - 1))
    davies_bouldin_score as dbi,         # DBI = (1/k) * sum(max((s(i) + s(j)) / d(c(i), c(j))))
    f1_score as f1,                      # F1 = (2 * TP) / (2 * TP + FP + FN)
    fowlkes_mallows_score as fmi,        # FMI = TP / sqrt((TP + FP) * (TP + FN))
    normalized_mutual_info_score as nmi, # NMI = (MI - Expected_MI) / sqrt(H(U) * H(V))
    precision_score as precision,        # Precision = TP / (TP + FP)
    recall_score as recall,              # Recall = TP / (TP + FN)
    roc_auc_score as auc,                # AUC = (TPR + TNR) / 2
)

from .hungarian import hungarian
from .metrics import conductance, modularity, p4


def evaluate(
    y_true: list,
    y_pred: list,
    x: Optional[list] = None,
    adj: Optional[spmatrix] = None,
    average: str = "macro",
    average_method: str = "arithmetic",
    multi_class: str = "ovr",
    metrics: Optional[list] = ["acc", "ami", "ari", "f1", "p4", "precision", "nmi", "recall"],
) -> dict:
    """
    Evaluate performance of clustering algorithms, including the following metrics:

    - 'acc': Accuracy or proportion of correct assignments.
    - 'ami': Adjusted Mutual Information, considering chance.
    - 'ari': Adjusted Rand Index, considering chance.
    - 'auc': Area Under the ROC (Receiver Operating Characteristic) Curve.
    - 'chi': Calinski-Harabasz Index, also known as Variance Ratio Criterion.
        Takes the ratio of the sum of between-clusters dispersion to within-cluster dispersion,
    - 'dbi': Davies Bouldin Score, avg. similarity of each cluster with its most similar cluster.
        Takes the ratio of the within-cluster distances to the between-cluster distances.
    - 'f1': F1 score, the harmonic mean of precision and recall.
    - 'fmi': Fowlkes-Mallows Index, geometric mean of precision and recall.
    - 'nmi': Normalized Mutual Information, considering entropy.
    - 'p4': P4 metric, probabilistic approach to F1 score.
    - 'precision': Positive predictive value.
    - 'recall': True positive rate (sensitivity/hit rate).

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :param x: Embeddings. Required for CHI and DBI.
    :param average: Average method for precision, recall, and F1 score.
        Available options: 'micro', 'macro', 'weighted', 'samples', 'binary', or None
        (returns values for multilabel classification). Default: 'macro'.
    :param average_method: Average method for NMI and AMI.
        Available options: 'min', 'geometric', 'arithmetic', 'max'. Default: 'arithmetic'.
    :param multi_class: Strategy to evaluate multiclass classification for ROC AUC only.
        Available options: 'ovo' (one-vs-one) or 'ovr' (one-vs-rest). Default: 'ovo'.
    :param metrics: List of metrics to compute. By default, computes
        'acc', 'ami', 'ari', 'f1', 'p4', 'precision' and 'recall'.
    """
    eva = {}

    assert x is not None or all(m not in metrics for m in ("chi", "dbi")),\
        "Metrics 'chi' and 'dbi' require embeddings `x` to be computed."
    assert adj is not None or all(m not in metrics for m in ("conductance", "modularity")),\
        "Metrics 'conductance' and 'modularity' require adjacency `adj` to be computed."
    assert len(y_true) == len(y_pred),\
        "Length of y_true and y_pred must be equal."

    y_true, y_pred = hungarian(y_true, y_pred)

    for metric in metrics:
        inp = (
            x if metric in ("chi", "dbi")
            else adj if metric in ("conductance", "modularity")
            else y_true
        )
        eva[metric] = globals()[metric](
            inp,
            y_pred,
            **({"average": average} if metric in ("precision", "recall", "f1", "auc") else {}),
            **({"average_method": average_method} if metric in ("nmi", "ami") else {}),
            **({"multi_class": multi_class} if metric == "auc" else {}),
        )

    return {k: float(v) for k, v in eva.items()}
