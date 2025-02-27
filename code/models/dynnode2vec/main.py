#!/usr/bin/env python3

import logging as log
import random
import sys
from argparse import ArgumentParser, Namespace

import networkx as nx
import networkx_temporal as tx
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from .graph import get_delta_nodes
from ..node2vec.cpu import Node2Vec


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

    parser.add_argument("--num-negative-samples",
                        help="Number of negative samples.",
                        default=1,
                        type=int)

    parser.add_argument("--epochs",
                        help="Number of epochs to set as maximum.",
                        default=1,
                        type=int)

    parser.add_argument("--lr",
                        help="Learning rate. Default: 0.025.",
                        default=0.025,
                        type=float)

    parser.add_argument("--p",
                        default=1.,
                        help="Higher values promote BFS. Default: 1.0.",
                        type=float)

    parser.add_argument("--q",
                        default=1.,
                        help="Lower values promote DFS. Default: 1.0.",
                        type=float)

    parser.add_argument("--c",
                        default=1.,
                        help="Lower values promote cycles. Default: 1.0.",
                        type=float)

    parser.add_argument("--r",
                        default=1.,
                        help="Lower values promote graph exploration. Default: 1.0.",
                        type=float)

    parser.add_argument("--t",
                        default=0,
                        help="Lower values promote temporal exploration. Default: 0.",
                        type=float)

    parser.add_argument("--directed",
                        action="store_true",
                        help="Set walks to respect edge direction.")

    parser.add_argument("--num-workers",
                        default=1,
                        help="Number of workers. Default: 1",
                        type=int)

    parser.add_argument("--softmax",
                        action="store_true",
                        dest="norm_exp",
                        help="Employ softmax to normalize probabilities.")

    parser.add_argument("--seed",
                        help="Random seed number(s). Optional.",
                        type=int)

    parser.add_argument("--weights",
                        default="weights_dynnode2vec",
                        help="File path to save or load weights.")

    parser.add_argument("--no-multi",
                        action="store_false",
                        dest="to_multi",
                        help="Do not allow multiple edges among nodes.")

    parser.add_argument("--bins",
                        action="store",
                        type=int,
                        help="Number of temporal graph snapshots to consider (optional).")

    parser.add_argument("--last-snapshot",
                        action="store_false",
                        dest="all_snapshots",
                        help="Store only last snapshot embedding from DynNode2Vec.")

    parser.add_argument("--merge-snapshots",
                        action="store_true",
                        help="Merge snapshots sequentially (no node/edge deletions).")

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    data = args.data

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    graphs = [
        to_networkx(
            data,
            edge_attrs=["time"],
            to_multi=args.to_multi,
        )
    ]
    if not args.directed:
        graphs[0] = graphs[0].to_undirected(as_view=False)

    graphs = tx.from_static(graphs[0])\
               .slice(bins=args.bins, attr="time")\
               .to_snapshots()

    model = Node2Vec(
        embedding_dim=args.embedding_dim,
        walks_per_node=args.walks_per_node,
        walk_length=args.walk_length,
        context_size=args.context_size,
        num_negative_samples=args.num_negative_samples,
        epochs=args.epochs,
        lr=args.lr,
        p=args.p,
        q=args.q,
        c=args.c,
        r=args.r,
        t=args.t,
        norm_exp=args.norm_exp,
        num_workers=args.num_workers,
        temporal=True,
        seed=args.seed,
    )

    for i, graph in enumerate(graphs):
        log.info(f"\nProcessing graph {i+1}/{len(graphs)}...")

        if i == 0:
            model(graph)
        else:
            graph = nx.compose_all(graphs[:i+1]) if args.merge_snapshots else graph
            walks = model.walk(graph, nodes=get_delta_nodes(graph, graphs[i-1]))
            model.word2vec.build_vocab(walks, update=True)
            model.word2vec.train(walks, total_examples=model.word2vec.corpus_count, epochs=model.epochs)

        if args.output and (i+1 == len(graphs) or args.all_snapshots):
            model.word2vec.wv.save_word2vec_format(
                f"{args.output}/feature{f'.{i}' if len(graphs) > 1 else ''}.emb"
            )


if __name__ == "__main__":
    sys.exit(main(getargs()))
