from typing import Union

from sklearn.cluster import SpectralClustering as SC
from numpy import ndarray
from torch import Tensor


class SpectralClustering():
    """
    Spectral clustering algorithm.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

    :param n_clusters: The dimension of the projection subspace.
    :param affinity: How to construct the affinity matrix.
    :param assign_labels: The strategy to use to assign labels in the embedding space.
    :param coef0: Kernel coefficient for poly kernels. Ignored by other kernels.
    :param degree: Degree of the polynomial kernel. Ignored by other kernels.
    :param eigen_solver: The eigenvalue decomposition strategy to use.
    :param eigen_tol: Stopping criterion for eigendecomposition of the Laplacian matrix.
    :param gamma: Kernel coefficient for rbf and poly kernels. Ignored by other kernels.
    :param kernel_params: Parameters of the kernel function.
    :param n_components: Number of eigenvectors to use for the spectral embedding.
    :param n_init: Number of time the k-means algorithm will be run with
        different centroid seeds, in case 'assign_labels' is 'kmeans'.
    :param n_jobs: The number of parallel jobs to run for neighbors search.
    :param n_neighbors: Number of neighbors to consider when constructing the
        affinity matrix using the nearest neighbors method.
    :param random_state: For predictable randomness.
    """
    def __init__(
        self,
        n_clusters: int,
        affinity: str = "precomputed",
        assign_labels: str = "cluster_qr",
        coef0: int = 1,
        degree: int = 3,
        eigen_solver: str = "lobpcg",
        eigen_tol: float = "auto",
        gamma: float = 1.0,
        kernel_params=None,
        n_components: int = None,
        n_init: int = 10,
        n_jobs: int = -1,
        n_neighbors: int = 10,
        random_state: int = None,
    ):
        self.sc = SC(
            n_clusters=n_clusters,
            affinity=affinity,
            assign_labels=assign_labels,
            coef0=coef0,
            degree=degree,
            eigen_solver=eigen_solver,
            eigen_tol=eigen_tol,
            gamma=gamma,
            kernel_params=kernel_params,
            n_components=n_components,
            n_init=n_init,
            n_jobs=n_jobs,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

    def fit(self, z: Union[list, ndarray, Tensor]):
        """
        :param z: Input data.
        """
        self.sc.fit(z)
        return self

    def fit_predict(self, z: Union[list, ndarray, Tensor]) -> list:
        """
        :param z: Input data.
        """
        return self.fit(z).labels_

    @property
    def labels_(self) -> list:
        return self.sc.labels_
