import numpy as np


def laplacian(A: np.ndarray, normalized=True) -> np.ndarray:
    L = A.copy()
    d = A.sum(axis=0)

    if normalized:
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if i == j:
                    L[i, j] = 1 if d[i] != 0 else 0
                else:
                    L[i, j] = - (1 / ((d[i] * d[j]) ** 0.5)) if A[i, j] == 1 else 0

    else:
        L[A > 0] = -1
        for i in range(A.shape[0]):
            L[i, i] = d[i]

    return L
