#!/usr/bin/env python3

import random
import sys
from argparse import ArgumentParser, Namespace

import keras
import numpy as np
import stellargraph as sg
import tensorflow as tf
import torch
from tensorflow import keras
from torch_geometric.utils.convert import to_networkx

from .attri2vec import attri2vec


def main(args: Namespace) -> None:
    data = args.data

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        keras.utils.set_random_seed(args.seed)

    SG = sg.StellarGraph.from_networkx(
        to_networkx(data, node_attrs=["x"], to_multi=args.to_multi).to_undirected(),
        node_features="x",
    )

    H = attri2vec(
        SG,
        epochs=args.epochs,
        number_of_walks=args.number_of_walks,
        length=args.length,
        layer_sizes=args.layer_sizes,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    np.save(args.output, H)


def getargs(argv: list = sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--epochs",
                        default=1,
                        help="Number of epochs to set as maximum.",
                        type=int)

    parser.add_argument("--lr",
                        help="Learning rate.",
                        default=1e-3,
                        type=float)

    parser.add_argument("--embedding-dim",
                        help="Embedding dimensions.",
                        default=[128],
                        dest="layer_sizes",
                        type=lambda x: x.split(","))

    parser.add_argument("--walk-length",
                        help="Walk length.",
                        default=80,
                        dest="number_of_walks",
                        type=int)

    parser.add_argument("--context-size",
                        help="Context size.",
                        default=10,
                        dest="length",
                        type=int)

    parser.add_argument("--walks-per-node",
                        help="Walks per node.",
                        default=10,
                        type=int)

    parser.add_argument("--batch_size",
                        default=20000,
                        help="Batch size to use.",
                        type=int)

    parser.add_argument("--workers",
                        default=1,
                        dest="num_workers",
                        help="Number of workers. Default: 1",
                        type=int)

    parser.add_argument("--no-multigraph",
                        action="store_false",
                        dest="to_multi",
                        help="Do not allow multiple edges among nodes.")

    parser.add_argument("--seed",
                        help="Random seed number.",
                        type=int)

    parser.add_argument("--output",
                        default="attri2vec_embeddings.npy",
                        metavar="FILE_PATH",
                        help="File path to save embeddings.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(getargs()))
