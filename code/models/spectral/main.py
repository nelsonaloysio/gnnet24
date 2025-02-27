#!/usr/bin/env python

import logging as log
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import torch
from torch_geometric.utils import to_dense_adj

from .spectral import SpectralClustering
from ..utils.evaluate import evaluate


def getargs(argv: list = sys.argv[1:]) -> Tuple[Namespace, list]:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--seed",
                        help="Random seed number.",
                        type=int)

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    data = args.data
    y = data.y.numpy()
    n_clusters = len(data.y.unique())
    A = to_dense_adj(data.edge_index)[0].numpy()
    sc = SpectralClustering(n_clusters=n_clusters, random_state=args.seed)
    y_pred = sc.fit_predict(A)
    eva = evaluate(y, y_pred)
    log.info(eva)


if __name__ == "__main__":
    sys.exit(main(getargs()))
