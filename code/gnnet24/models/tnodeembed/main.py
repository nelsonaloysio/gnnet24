#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser, Namespace

import torch
from torch_geometric.utils.convert import to_networkx

from .tnodeembed import tNodeEmbed


def main(args: Namespace) -> None:
    data = args.data

    G = to_networkx(data, edge_attrs=["time"], to_multi=True)

    if args.dump_folder is not None:
        os.makedirs(args.dump_folder, exist_ok=True)

    model = tNodeEmbed(
        G,
        task="node_classification",
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
        align=args.align,
        dump_folder=args.dump_folder,
    )

    time = {}
    with open(args.output, "w") as f:
        f.write(f"{model.graph_nx.order()} {args.embedding_dim}\n")
        # Save embeddings from nodes in all time steps.
        for node, x in model.graph_nx.nodes(data=True):
            f.write(f"{node} {' '.join(map(str, next(reversed(x.values()))))}\n")
            # Store nodes per time step.
            if args.output_snapshots:
                for t in list(x):
                    time[t] = time.get(t, []) + [node]

    if args.output_snapshots:
        filename = os.path.splitext(args.output)[0]
        for t in sorted(time):
            timestep = int(t) if int(t) == t else t
            # Save embeddings from nodes in current time step only.
            with open(f"{filename}_t={timestep}.emb", "w") as f:
                f.write(f"{len(time[t])} {args.embedding_dim}\n")
                for node in time[t]:
                    f.write(f"{node} {' '.join(map(str, model.graph_nx.nodes[node][t]))}\n")


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

    parser.add_argument("--no-align",
                        action="store_false",
                        dest="align",
                        help="Whether to disable alignment of node features.")

    parser.add_argument("--output",
                        default="tnodeembed_embeddings.emb",
                        metavar="FILE_PATH",
                        help="File path to save embeddings.")

    parser.add_argument("--output-snapshots",
                        action="store_true",
                        help="Save embeddings from all graph snapshots.")

    parser.add_argument("--dump-folder",
                        metavar="FOLDER_PATH",
                        help="Folder path to save temporary files for later use. Optional.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(getargs()))
