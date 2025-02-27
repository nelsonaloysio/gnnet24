#!/usr/bin/env python3

import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx

from .models import tNodeEmbed


def getargs(argv: list = sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILEPATH",
                        type=torch.load)

    parser.add_argument("--weights",
                        default="weights_tnodeembed",
                        help="File path to save or load weights.")

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    data = args.data

    G = to_networkx(data, edge_attrs=["time"], to_multi=True)
    model = tNodeEmbed(G, task="node_classification", dump_folder=args.weights)

    X = np.array([
        list(G.node[node]["x"])
        for node in sorted(model.graph_nx.nodes())
    ])

    np.save(f"{args.weights}.npy", X)


if __name__ == "__main__":
    sys.exit(main(getargs()))
