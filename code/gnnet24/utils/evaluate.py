from typing import Optional

import numpy as np
from munkres import Munkres
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score as acc,             # ACC = (TP + TN) / (TP + TN + FP + FN)
    adjusted_mutual_info_score as ami, # AMI = (MI - Expected_MI) / (max(H(U), H(V)) - Expected_MI)
    adjusted_rand_score as ari,        # ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    calinski_harabasz_score as chi,    # CHI = (B / W) * ((n - k) / (k - 1))
    davies_bouldin_score as dbi,       # DBI = (1/k) * sum(max((s(i) + s(j)) / d(c(i), c(j))))
    f1_score as f1,                    # F1 = (2 * TP) / (2 * TP + FP + FN)
    fowlkes_mallows_score as fmi,      # FMI = TP / sqrt((TP + FP) * (TP + FN))
    normalized_mutual_info_score as nmi, # NMI = (MI - Expected_MI) / sqrt(H(U) * H(V))
    precision_score as precision,      # Precision = TP / (TP + FP)
    recall_score as recall,            # Recall = TP / (TP + FN)
    roc_auc_score as auc,              # AUC = (TPR + TNR) / 2
)
# from scipy.optimize import linear_sum_assignment


def evaluate(
    y_true: list,
    y_pred: list,
    x: Optional[list] = None,
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
    y_true, y_pred = hungarian(y_true, y_pred)

    eva = {}

    assert x is not None or all(m not in metrics for m in ("chi", "dbi")),\
        "Metrics 'chi' and 'dbi' require embeddings `x` to be computed."

    # Compute using sklearn.metrics.
    for metric in metrics:
        eva[metric] = globals()[metric](
            x if metric in ("chi", "dbi") else y_true,
            y_pred,
            **({"average": average} if metric in ("precision", "recall", "f1", "auc") else {}),
            **({"average_method": average_method} if metric in ("nmi", "ami") else {}),
            **({"multi_class": multi_class} if metric == "auc" else {}),
        )

    return {k: float(v) for k, v in eva.items()}


def hungarian(y_true: list, y_pred: list):
    """
    Kuhn-Munkres algorithm (Hungarian method) to solve cluster label assignments.

    Original implementation:
        https://github.com/karenlatong/AGC-master/blob/master/metrics.py

    - Step 0: Create an nxm matrix called the cost matrix in which each element
        represents the cost of assigning one of n workers to one of m jobs.
        Rotate the matrix so that there are at least as many columns as rows and
        let k=min(n,m).

    - Step 1: For each row of the matrix, find the smallest element and subtract
        it from every element in its row. Go to Step 2.

    - Step 2: Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the matrix.

    - Step 3: Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE; otherwise, Go to Step 4.

    - Step 4: Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise, cover
        this row and uncover the column containing the starred zero. Continue in
        this manner until there are no uncovered zeros left. Save the smallest
        uncovered value and Go to Step 6.

    - Step 5: Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4,
        Z1 denote the starred zero in the column of Z0 (if any), and Z2 denote
        the primed zero in the row of Z1 (there will always be one). Continue
        until the series terminates at a primed zero that has no starred zero in
        its column. Unstar each starred zero of the series, star each primed
        zero of the series, erase all primes and uncover every line in the
        matrix. Return to Step 3.

    - Step 6: Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column. Return
        to Step 4 without altering any stars, primes, or covered lines.

    - DONE: Assignment pairs are indicated by the positions of the starred zeros
        in the cost matrix. If C(i,j) is a starred zero, the element associated
        with row i is assigned to the element associated with column j.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    """
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # Match two clustering results by Munkres algorithm.
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # Get the match results.
    new_pred = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # Correponding label in l2.
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the pred_label list.
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_pred[ai] = c

    return y_true, new_pred


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
