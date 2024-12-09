#!/usr/bin/env python3

import sys
from argparse import ArgumentParser, Namespace

import igraph as ig
import torch
from torch_geometric.utils.convert import to_networkx

from .leiden import leiden, OPT


def main(args: Namespace) -> None:
    data = args.data

    n_clusters = len(data.y.numpy().unique())

    iG = ig.Graph.from_networkx(
        to_networkx(
            data,
            to_multi=True,
        )
    )

    leiden(
        graph=iG,
        n_clusters=n_clusters,
        opt=args.opt,
        resolution=args.resolution,
        iterations=args.iterations,
        seed=args.seed,
    )


def getargs(argv: list = sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--optimizer", "--optimiser",
                        choices=list(OPT),
                        default="modularity",
                        dest="opt",
                        help="Leiden algorithm optimizer. Default: 'modularity'.",
                        type=lambda x: OPT[x.lower()])

    parser.add_argument("--resolution",
                        help="Resolution parameter. Optional.",
                        type=float)

    parser.add_argument("--iterations",
                        help="Number of iterations. Optional.",
                        type=int)

    parser.add_argument("--seed",
                        help="Random seed number. Optional.",
                        type=int)

    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(getargs()))
