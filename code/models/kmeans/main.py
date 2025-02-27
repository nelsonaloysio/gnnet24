#!/usr/bin/env python

import logging as log
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import torch

from .kmeans import KMeans
from ..utils.evaluate import evaluate


def getargs(argv: list = sys.argv[1:]) -> Tuple[Namespace, list]:
    """ Get arguments from command line. """
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--seed",
                        help="Random seed number.",
                        type=int)

    parser.add_argument("--split",
                        action="store_true",
                        help="Split data into train and test sets.")

    parser.add_argument("--device",
                        choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model.")

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    """ Main function. """
    data = args.data

    y = data.y.numpy()
    if args.split:
        y = y[data.test_mask]

    y_pred = KMeans(
        n_clusters=len(data.y.unique()),
        seed=args.seed,
        device=args.device,
    )\
    .fit(data.x[data.train_mask] if args.split else data.x)\
    .predict(data.x[data.test_mask] if args.split else data.x)

    eva = evaluate(y, y_pred)
    log.info(eva)


if __name__ == "__main__":
    sys.exit(main(getargs()))
