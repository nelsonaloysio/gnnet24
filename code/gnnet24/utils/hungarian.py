import numpy as np
from munkres import Munkres
# from scipy.optimize import linear_sum_assignment


def hungarian(y_true: list, y_pred: list):
    """
    Kuhn-Munkres algorithm (Hungarian method) to solve cluster label assignments.

    Original implementation from:
    https://github.com/karenlatong/AGC-master/blob/master/metrics.py

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    """
    y_pred = dict(zip(np.unique(y_pred), range(len(np.unique(y_pred)))))
    y_pred = [y_pred[i] for i in y_pred.keys()]

    y_true = dict(zip(np.unique(y_true), range(len(np.unique(y_true)))))
    y_true = [y_true[i] for i in y_true.keys()]

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        for l in l1:
            if l not in l2:
                l2.append(l)
        for l in l2:
            if l not in l1:
                l1.append(l)
        assert len(l1) == len(l2), "The number of classes do not match."
        numclass1, numclass2 = len(l1), len(l2)

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
