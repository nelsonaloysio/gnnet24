#!/usr/bin/env python3

import random
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from .cpu import Node2Vec as Node2VecCPU
from .gpu import Node2Vec as Node2VecGPU


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
                        default="weights_node2vec",
                        help="File path to save or load weights.")

    parser.add_argument("--device",
                        choices=["cpu", "gpu"],
                        default="cpu",
                        help="Device to run the model. Default: 'cpu'.")

    parser.add_argument("--no-multi",
                        action="store_false",
                        dest="to_multi",
                        help="Do not allow multiple edges among nodes.")

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    data = args.data

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.device == "cuda":
        assert not args.temporal, "TemporalNode2Vec is not implemented on GPU."

        model = Node2VecGPU(data, **args)

        for epoch in range(args.epochs):
            # Backward pass.
            loss = 0
            for pos_rw, neg_rw in model.loader:
                model.optimizer.zero_grad()
                loss += (loss_ := model.loss(pos_rw.to(model.device), neg_rw.to(model.device))).item()
                loss_.backward()
                model.optimizer.step()
            loss /= len(model.loader)

            # Forward pass.
            z = model().detach().cpu().numpy()

            # if es(loss):
            #     break
            # if es.best_epoch == epoch:
            #     np.save(args.output, z)

        # z = np.load(args.output)

    else:
        G = to_networkx(data, to_multi=args.to_multi)
        if not args.directed:
            G = G.to_undirected(as_view=False)

        model = Node2VecCPU(
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
            norm_exp=args.norm_exp,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        model(G)
        model.word2vec.wv.save_word2vec_format(f"{args.weights}.emb")

if __name__ == "__main__":
    sys.exit(main(getargs()))
