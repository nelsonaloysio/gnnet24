from typing import Literal, Optional, Union
from warnings import warn

import torch
from numpy import ndarray
from sklearn.cluster import KMeans as KMeansCPU
from torch import Tensor

try:
    from torch_kmeans import (
        KMeans as KMeansCUDA,
        SoftKMeans as SoftKMeansCUDA,
    )
    from torch_kmeans.utils.distances import (
        CosineSimilarity,
        DotProductSimilarity,
        LpDistance,
    )
    DISTANCES = {
        "l2": LpDistance,
        "l1": LpDistance,
        "cosine": CosineSimilarity,
        "dotproduct": DotProductSimilarity,
    }
except:
    KMeansCUDA = None
    DISTANCES = {}


class KMeans():
    """
    K-means clustering algorithm using either CPU or GPU.

    - CPU implementation uses scikit-learn:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    - GPU implementation uses torch_kmeans:
        https://torch-kmeans.readthedocs.io/en/latest/

    :param n_clusters: Number of clusters, as well as the number of centroids.
    :param init: Method for initialization of centroids (default: 'k-means++').
        - 'k-means++': selects initial cluster centers for k-mean clustering
            in a smart way to speed up convergence.
        - 'random': choose k observations (rows) at random from data for the
            initial centroids.
    :param n_init: Number of time the k-means algorithm will be run with
        different centroid seeds (default: 20). The final results will be the
        best output of `n_init` consecutive runs in terms of inertia.
    :param max_iter: Maximum number of iterations of the k-means algorithm
        for a single run.
    :param distance: Distance function to use for clustering.
        - 'l2': Euclidean distance (default).
        - 'l1': Manhattan distance.
        - 'cosine': Cosine similarity.
        - 'dotproduct': Dot product similarity.
    :param tol: Relative tolerance with regards to inertia to declare convergence.
        Default: 1e-4.
    :param soft: If True, use soft K-Means, as implemented by ClusterNet (CUDA only).
        Reference: https://arxiv.org/pdf/1905.13732.pdf
    :param device: Device to use for computation.
        Default: 'cuda' if available; otherwise, 'cpu'.
    :param seed: Random seed to use (optional).
    :param verbose: Verbosity mode. If True, progress of the algorithm is printed.
    """
    def __init__(
        self,
        n_clusters: int,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 20,
        max_iter: int = 300,
        distance: Optional[Literal["l2", "l1", "cosine", "dotproduct"]] = None,
        tol: float = 1e-4,
        soft: bool = False,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":

            if distance is not None:
                warn("Custom distance functions are only supported in the CUDA implementation. "
                     "Defaulting to Euclidean ('l2') distance.")

            if soft is True:
                warn("Soft K-Means algorithm is only supported in the CUDA implementation. "
                     "Defaulting to regular K-Means (sklearn) algorithm.")

            self.kmeans = KMeansCPU(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
                verbose=1 if verbose else 0
            )

        elif self.device == "cuda":
            if KMeansCUDA is None:
                raise ModuleNotFoundError("Package 'torch_kmeans' not found.")

            if distance is None:
                distance = "cosine" if soft else "l2"

            assert distance in DISTANCES,\
                   "Invalid distance function, must be either "\
                   "'l2', 'l1', 'cosine', or 'dotproduct'."

            self.kmeans = (SoftKMeansCUDA if soft else KMeansCUDA)(
                n_clusters=n_clusters,
                init=init,
                num_init=n_init,
                max_iter=max_iter,
                distance=DISTANCES[distance],
                p_norm=2 if distance == "l2" else 1,
                tol=tol,
                seed=seed,
                verbose=verbose,
            )

        else:
            raise AssertionError("Invalid device set, please use either 'cpu' or 'cuda'.")

    def fit(self, z: Union[list, ndarray, Tensor]):
        """
        Compute k-means clustering.

        Note that this method does not store the results in case
        of the GPU implementaiton. Use `fit_predict` instead.

        :param z: Input data.
        """
        z = self.__z(z)
        if self.device == "cpu":
            self.kmeans.fit(z.data.cpu().numpy())
        else:
            self.kmeans.fit(z.view(1, z.shape[0], z.shape[1]))
        return self

    def fit_predict(self, z: Union[list, ndarray, Tensor]) -> list:
        """
        Compute cluster centers and predict cluster index for each sample.

        :param z: Input data.
        """
        z = self.__z(z)
        if self.device == "cpu":
            return self.kmeans.fit_predict(z.data.cpu().numpy())
        return self.kmeans\
            .fit_predict(z.view(1, z.shape[0], z.shape[1]))\
            .detach()\
            .data\
            .cpu()\
            .numpy()\
            .flatten()

    def predict(self, z: Union[list, ndarray, Tensor]) -> list:
        """
        Predict the closest cluster each sample in z belongs to.

        :param z: Input data.
        """
        z = self.__z(z)
        if self.device == "cpu":
            return self.kmeans.predict(z.data.cpu().numpy())
        return self.kmeans\
            .predict(z.view(1, z.shape[0], z.shape[1]))\
            .detach()\
            .data\
            .cpu()\
            .numpy()\
            .flatten()

    @property
    def cluster_centers_(self) -> Tensor:
        """ Return coordinates of current centroids. """
        if self.device == "cpu":
            return self.kmeans.cluster_centers_
        return self.kmeans._result.centers

    @property
    def labels_(self) -> ndarray:
        """ Return labels of each point. """
        if self.device == "cpu":
            return self.kmeans.labels_
        return self.kmeans._result.labels.cpu().numpy().flatten()

    def __z(self, z: Union[list, ndarray, Tensor]) -> Tensor:
        """ Convert input data to Tensor. """
        if isinstance(z, list):
            z = torch.tensor(z)
        elif isinstance(z, ndarray):
            z = torch.from_numpy(z)
        return z.detach().to(self.device)
