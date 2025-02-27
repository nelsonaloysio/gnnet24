#!/usr/bin/env python3

import logging as log
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import stellargraph as sg
import torch
from torch_geometric.utils.convert import to_networkx

from .attri2vec import attri2vec


def getargs(argv: list = sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--seed",
                        help="Random seed number.",
                        type=int)

    parser.add_argument("--weights",
                        default="weights_attri2vec",
                        help="File path to save or load weights.")

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    data = args.data

    SG = sg.StellarGraph.from_networkx(
        to_networkx(data, node_attrs=["x"], to_multi=False),
        node_features="x"
    )

    H = attri2vec(SG, seed=args.seed)
    np.save(f"{args.weights}_seed={args.seed}.npy", H)
    return H


if __name__ == "__main__":
    sys.exit(main(getargs()))
