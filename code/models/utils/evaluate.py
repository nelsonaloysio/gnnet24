from typing import Optional

import numpy as np
from munkres import Munkres
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
# from scipy.optimize import linear_sum_assignment


def evaluate(
    y_true: list,
    y_pred: list,
    x: Optional[list] = None,
    average: str = "macro",
    average_method: str = "arithmetic"
) -> dict:
    """
    Evaluate clustering performance.
    """
    y_true, y_pred = hungarian(y_true, y_pred)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "nmi": normalized_mutual_info_score(y_true, y_pred, average_method=average_method),
        "ami": adjusted_mutual_info_score(y_true, y_pred, average_method=average_method),
        "fmi": fowlkes_mallows_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "p4": p4_metric(y_true, y_pred),
    }

    if x is not None:
        metrics.update({
            "chi": calinski_harabasz_score(x, y_pred),
            "dbi": davies_bouldin_score(x, y_pred),
        })

    return {k: float(v) for k, v in metrics.items()}


def hungarian(y_true: list, y_pred: list):
    """
    Kuhn-Munkres algorithm (Hungarian method) to solve cluster label assignments.

    Original implementation:
        - https://github.com/karenlatong/AGC-master/blob/master/metrics.py

    - Step 0: Create an nxm matrix called the cost matrix in which each elementrepresents the cost
        of assigning one of n workers to one of m jobs. Rotate the matrix so that there are at least
        as many columns as rows and let k=min(n,m).

    - Step 1: For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.

    - Step 2: Find a zero (Z) in the resulting matrix. If there is no starred zero in
        its row or column, star Z. Repeat for each element in the matrix. Go to Step 3.

    - Step 3: Cover each column containing a starred zero. If K columns are covered, the starred
        zeros describe a complete set of unique assignments. In this case, Go to DONE; otherwise,
        Go to Step 4.

    - Step 4: Find a noncovered zero and prime it. If there is no starred zero in the row containing
        this primed zero, Go to Step 5. Otherwise, cover this row and uncover the column containing
        the starred zero. Continue in this manner until there are no uncovered zeros left. Save the
        smallest uncovered value and Go to Step 6.

    - Step 5: Construct a series of alternating primed and starred zeros as follows. Let Z0
        represent the uncovered primed zero found in Step 4. Let Z1 denote the starred zero in the
        column of Z0 (if any). Let Z2 denote the primed zero in the row of Z1 (there will always be
        one). Continue until the series terminates at a primed zero that has no starred zero in its
        column. Unstar each starred zero of the series, star each primed zero of the series, erase
        all primes and uncover every line in the matrix. Return to Step 3.

    - Step 6: Add the value found in Step 4 to every element of each covered row,
        and subtract it from every element of each uncovered column. Return to Step 4
        without altering any stars, primes, or covered lines.

    - DONE: Assignment pairs are indicated by the positions of the starred zeros
        in the cost matrix. If C(i,j) is a starred zero, then the element associated
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


def p4_metric(y_true: list, y_pred: list):
    """
    Calculate the P4 metric.

    Reference:
        - https://arxiv.org/abs/2210.11997

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
