#!/usr/bin/env python3

import logging as log
import random
import sys
from argparse import ArgumentParser, Namespace

import igraph as ig
import leidenalg as la
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from ..utils.evaluate import evaluate

OPTS = {
    "modularity": la.ModularityVertexPartition,
    "cpm": la.CPMVertexPartition,
    "rb_pots": la.RBConfigurationVertexPartition,
    "rber_pots": la.RBERVertexPartition,
    "significance": la.SignificanceVertexPartition,
    "surprise": la.SurpriseVertexPartition,
}


def getargs(argv: list = sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--optimizer", "--optimiser",
                        choices=["modularity", "cpm", "rb_pots", "rber_pots", "significance", "surprise"],
                        default="modularity",
                        dest="opt",
                        help="Leiden algorithm optimizer. Default: 'modularity'.",
                        type=lambda x: OPTS[x.lower()])

    parser.add_argument("--iterations",
                        help="Number of iterations. Optional..",
                        type=int)

    parser.add_argument("--resolution",
                        help="Resolution parameter. Optional.",
                        type=float)

    parser.add_argument("--seeds",
                        help="Random seed number. Optional.",
                        type=lambda x: [int(x) for x in x.split(",")])

    parser.add_argument("--runs",
                        default=1,
                        help="Number of runs, if seeds are not given. Default: 1.",
                        type=int)

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    data = args.data

    y = data.y.numpy()
    n_clusters = len(data.y.unique())

    G = ig.Graph.from_networkx(
        to_networkx(
            data,
            to_multi=True,
        )
    )

    args_partition = dict(
        initial_membership=np.random.choice(n_clusters, G.vcount()),
        resolution_parameter=args.resolution,
    )

    args_opt = dict(
        n_iterations=args.iterations,
    )

    for i in range(len(args.seeds) if args.seeds else args.runs):
        seed = args.seeds[i] if args.seeds else None

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        partition = args.opt(
            G,
            **{k: v for k, v in args_partition.items() if v is not None},
        )

        opt = la.Optimiser()
        opt.consider_empty_community = False
        opt.set_rng_seed = seed

        opt.optimise_partition(
            partition,
            **{k: v for k, v in args_opt.items() if v is not None},
        )

        log.info(
            "Leiden: Q=%.5f (clusters: %d, modules: %d)" % (
                partition.quality(),
                n_clusters,
                len(set(partition.membership))
            )
        )

        eva = evaluate(y, partition.membership)
        log.info(eva)


if __name__ == "__main__":
    sys.exit(main(getargs()))
