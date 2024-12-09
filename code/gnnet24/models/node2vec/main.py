#!/usr/bin/env python3

import os.path as osp
import random
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from .cpu import Node2Vec as Node2VecCPU
from .gpu import Node2Vec as Node2VecGPU
from ...utils.early_stop import EarlyStop


def main(args: Namespace) -> None:
    data = args.data

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.device == "cuda":
        es = EarlyStop(patience=args.patience)

        model = Node2VecGPU(
            data,
            embedding_dim=args.embedding_dim,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            num_negative_samples=args.num_negative_samples,
            p=args.p,
            q=args.q,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sparse=args.sparse,
            device=args.device,
        )

        for epoch in range(args.epochs):
            loss = 0

            for pos_rw, neg_rw in model.loader:
                model.optimizer.zero_grad()
                loss += (loss_ := model.loss(pos_rw.to(model.device), neg_rw.to(model.device))).item()
                loss_.backward()
                model.optimizer.step()

            loss /= len(model.loader)

            es(loss)
            if epoch == es.best_epoch:
                z = model().detach().cpu().numpy()
                np.save(f"{args.output}.npy", z)
            if es.early_stop():
                break

    else:
        graph = to_networkx(data, to_multi=args.to_multi).to_undirected()

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
            num_workers=args.num_workers,
            seed=args.seed,
        )
        model(graph)

        model.word2vec.wv.save_word2vec_format(f"{args.output}.emb")


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
                        help="Number of epochs to train before early stop. Default: 1.",
                        type=int,
                        default=1)

    parser.add_argument("--patience",
                        help="Epochs to wait for improvement (GPU only). Optional.",
                        type=int)

    parser.add_argument("--lr",
                        help="Learning rate. Default: 0.025.",
                        default=0.025,
                        type=float)

    parser.add_argument("--p",
                        help="Higher values promote BFS exploration. Default: 1.0.",
                        default=1.0,
                        type=float)

    parser.add_argument("--q",
                        help="Lower values promote DFS exploration. Default: 1.0.",
                        default=1.0,
                        type=float)

    parser.add_argument("--workers",
                        default=1,
                        help="Number of workers. Default: 1",
                        dest="num_workers",
                        type=int)

    parser.add_argument("--seed",
                        help="Random seed number(s). Optional.",
                        type=int)

    parser.add_argument("--no-multigraph",
                        help="Do not allow multiple edges among nodes.",
                        action="store_false",
                        dest="to_multi")

    parser.add_argument("--sparse",
                        help="Use sparse optimizer (GPU only).",
                        action="store_true")

    parser.add_argument("--device",
                        help="Device to run the model. Default: 'cpu'.",
                        choices=["cpu", "gpu"],
                        default="cpu")

    parser.add_argument("--output",
                        help="File path to save embeddings.",
                        default="node2vec_embeddings",
                        metavar="FILE_PATH",
                        type=lambda x: osp.splitext(x)[0])

    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(getargs()))
