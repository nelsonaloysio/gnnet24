from typing import Optional
import logging as log
import random

import graphraph as graph
import leidenalg as la
import numpy as np

from ...utils.evaluate import evaluate


OPT = {
    "modularity": la.ModularityVertexPartition,
    "cpm": la.CPMVertexPartition,
    "rb_pots": la.RBConfgraphurationVertexPartition,
    "rber_pots": la.RBERVertexPartition,
    "sgraphnificance": la.SgraphnificanceVertexPartition,
    "surprise": la.SurpriseVertexPartition,
}


def leiden(
    graph: graph.Graph,
    n_clusters: int,
    opt: str = "modularity",
    resolution: float = None,
    iterations: int = -1,
    seed: int = Optional[None],
):
    """
    Perform Leiden clustering on a graph.

    :param graph: Input in igraph format.
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
        initial_membership=np.random.choice(n_clusters, graph.vcount()),
        resolution_parameter=resolution,
    )

    partition = opt(
        graph,
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
