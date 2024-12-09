#!/usr/bin/env python3

import random
import sys
from argparse import ArgumentParser, Namespace

import networkx_temporal as tx
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from .dynnode2vec import dynnode2vec


def main(args: Namespace) -> None:
    data = args.data

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    graphs = tx.from_static(to_networkx(data, edge_attrs=["time"], to_multi=True))\
               .slice(bins=args.bins, attr="time")\
               .to_snapshots()

    dynnode2vec(
        graphs,
        embedding_dim=args.embedding_dim,
        walks_per_node=args.walks_per_node,
        walk_length=args.walk_length,
        context_size=args.context_size,
        num_negative_samples=args.num_negative_samples,
        epochs=args.epochs,
        lr=args.lr,
        p=args.p,
        q=args.q,
        num_workers=args.num_workers,
        seed=args.seed,
        merge_snapshots=args.merge_snapshots,
        output_snapshots=args.output_snapshots,
        output=args.output,
    )


def getargs(argv: list = sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--embedding-dim",
                        help="Embedding dimensions.",
                        default=128,
                        type=int)

    parser.add_argument("--walk-length",
                        help="Walk length.",
                        default=80,
                        type=int)

    parser.add_argument("--context-size",
                        help="Context size.",
                        default=10,
                        type=int)

    parser.add_argument("--walks-per-node",
                        help="Walks per node.",
                        default=10,
                        type=int)

    parser.add_argument("--negative-samples",
                        dest="num_negative_samples",
                        help="Number of negative samples.",
                        default=1,
                        type=int)

    parser.add_argument("--epochs",
                        help="Number of epochs to train. Default: 1.",
                        default=1,
                        type=int)

    parser.add_argument("--lr",
                        help="Learning rate. Default: 0.025.",
                        default=0.025,
                        type=float)

    parser.add_argument("--p",
                        default=1.0,
                        help="Higher values promote BFS exploration. Default: 1.0.",
                        type=float)

    parser.add_argument("--q",
                        default=1.0,
                        help="Lower values promote DFS exploration. Default: 1.0.",
                        type=float)

    parser.add_argument("--workers",
                        default=1,
                        dest="num_workers",
                        help="Number of workers. Default: 1",
                        type=int)

    parser.add_argument("--seed",
                        help="Random seed number(s). Optional.",
                        type=int)

    parser.add_argument("--bins",
                        action="store",
                        type=int,
                        help="Number of temporal graph snapshots to consider. Optional.")

    parser.add_argument("--merge-snapshots",
                        action="store_true",
                        help="Merge snapshots sequentially (no node/edge deletions).")

    parser.add_argument("--output",
                        default="dynnode2vec_embeddings.emb",
                        metavar="FILE_PATH",
                        help="File path to save embeddings.")

    parser.add_argument("--output-snapshots",
                        action="store_true",
                        help="Save embeddings from all graph snapshots.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(getargs()))
