import numpy as np
from munkres import Munkres
# from scipy.optimize import linear_sum_assignment


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
