from typing import Optional
import logging as log
import random

import igraph as ig
import leidenalg as la
import numpy as np

from ...utils.evaluate import evaluate

OPT = {
    "modularity": la.ModularityVertexPartition,
    "cpm": la.CPMVertexPartition,
    "rb_pots": la.RBConfigurationVertexPartition,
    "rber_pots": la.RBERVertexPartition,
    "significance": la.SignificanceVertexPartition,
    "surprise": la.SurpriseVertexPartition,
}


def leiden(
    iG: ig.Graph,
    n_clusters: int,
    opt: str = "modularity",
    iterations: int = -1,
    resolution: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Perform Leiden clustering on a igraph.

    :param iG: Input in igraph format.
    :param n_clusters: Number of clusters.
    :param opt: Leiden algorithm optimizer.
    :param resolution: Resolution parameter.
    :param iterations: Number of iterations.
    :param seed: Random seed number.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    args_partition = dict(
        initial_membership=np.random.choice(n_clusters, ig.vcount()),
        resolution_parameter=resolution,
    )

    partition = opt(
        iG,
        **{k: v for k, v in args_partition.items() if v is not None},
    )

    opt = la.Optimiser()
    opt.consider_empty_community = False
    opt.set_rng_seed = seed

    args_opt = dict(
        n_iterations=iterations,
    )

    opt.optimise_partition(
        partition,
        **{k: v for k, v in args_opt.items() if v is not None},
    )

    log.info(
        "Leiden: Q=%.5f (clusters: %d, modules: %d)",
        partition.quality(),
        n_clusters,
        len(set(partition.membership))
    )

    eva = evaluate(y, partition.membership)
    log.info(eva)

    return partition
