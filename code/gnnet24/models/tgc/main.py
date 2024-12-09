from argparse import ArgumentParser, Namespace
import sys

import numpy as np
import torch

from .tgc import TGC


def getargs(argv: list = sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Dataset name to load.",
                        metavar="NAME",
                        target="dataset")

    parser.add_argument("--epochs",
                        help="Number of epochs to train before early stop.",
                        type=int)

    parser.add_argument('--neg-size',
                        help="Number of negative samples.",
                        type=int)

    parser.add_argument('--hist-len',
                        help="Length of the history.",
                        type=int)

    parser.add_argument('--save-step',
                        help="Number of steps to save model weights.",
                        type=int)

    parser.add_argument('--batch-size',
                        help="Batch size.",
                        type=int)

    parser.add_argument('--lr',
                        help="Learning rate.",
                        type=float)

    parser.add_argument('--emb-size',
                        help="Embedding size.",
                        type=int)

    parser.add_argument('--seed',
                        help="Random seed number.",
                        type=int)

    parser.add_argument('--directed',
                        help="Consider edges as directed.",
                        action="store_true")

    parser.add_argument('--static',
                        help="Consider graph as static.",
                        action="store_true")

    parser.add_argument('--split',
                        choices=["transductive", "inductive"],
                        default=None,
                        help="Learning setting. If unset, the full dataset is used for training (no split).")

    parser.add_argument("--device",
                        help="Device to run the model.",
                        choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--device-kmeans",
                        help="Device to run the clusterer.",
                        choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args(argv)


def main(args: Namespace) -> None:
    if args.data == "dblp":
        args.epoch = 10 if args.epoch is None else args.epoch
        args.neg_size = 50 if args.neg_size is None else args.neg_size
        args.save_step = 5 if args.save_step is None else args.save_step
    else:
        args.epoch = 200 if args.epoch is None else args.epoch
        args.neg_size = 2 if args.neg_size is None else args.neg_size
        args.save_step = 50 if args.save_step is None else args.save_step

    args.hist_len = 1 if args.hist_len is None else args.hist_len
    args.batch_size = 1024 if args.batch_size is None else args.batch_size
    args.learning_rate = 0.001 if args.learning_rate is None else args.learning_rate
    args.emb_size = 16 if args.emb_size is None else args.emb_size

    tgc = TGC(args)
    tgc.train()


if __name__ == "__main__":
    sys.exit(main(getargs()))
