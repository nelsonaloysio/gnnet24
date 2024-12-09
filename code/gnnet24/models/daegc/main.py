import logging as log
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from typing import Tuple

import numpy as np
import torch

from .daegc import DAEGC
from .preprocess import preprocess
from .pretrain import pretrain
from .train import train


def main(args: Namespace) -> None:
    """ Main function. """
    data = args.data

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    data = preprocess(data, t=args.hops, device=args.device)

    model = DAEGC(
        in_features=data.num_features,
        hidden_size=args.hidden_size,
        out_features=args.out_features,
        n_clusters=len(np.unique(data.y)),
        alpha=args.alpha,
    ).to(args.device)

    os.makedirs(args.weights, exist_ok=True)
    weights_pretrain = os.path.join(args.weights, "pretrain.pkl")
    weights_train = os.path.join(args.weights, "train.pkl")

    # Pretrain model.
    if "pretrain" in args.stage.split("+"):
        log.info("Pretraining model...")
        pretrain(
            model=model,
            data=data,
            lr=args.lr or 5e-3,
            weight_decay=args.weight_decay,
            n_init=args.n_init,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            weights=weights_pretrain,
            device=args.device,
        )

    # Load pretrained weights.
    elif os.path.isfile(weights_pretrain):
        log.info("Loading weights from pretrain...")
        model.load_state_dict(torch.load(weights_pretrain, map_location="cpu"))

    # Train model.
    if "train" in args.stage.split("+"):
        log.info("Training model...")
        train(
            model=model,
            data=data,
            lr=args.lr or 1e-4,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            update_interval=args.update_interval,
            n_init=args.n_init,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            weights=weights_train,
            device=args.device,
        )


def getargs(argv: list = sys.argv[1:]) -> Tuple[Namespace, list]:
    """ Get arguments. """
    parser = ArgumentParser()

    parser.add_argument("--data",
                        help="Processed dataset to load.",
                        metavar="FILE_PATH",
                        type=torch.load)

    parser.add_argument("--hidden-size",
                        help="Hidden feature size. Default: 256.",
                        default=256,
                        type=int)

    parser.add_argument("--out-features",
                        help="Output feature size. Default: 16.",
                        default=16,
                        type=int)

    parser.add_argument("--lr",
                        help="Learning rate of the optimizer. "
                             "Default: 5e-3 for pretrain, 1e-4 for train.",
                        type=float)

    parser.add_argument("--weight-decay",
                        help="Weight decay of the optimizer. Default: 5e-3.",
                        type=float,
                        default=5e-3)

    parser.add_argument("--update-interval",
                        help="Interval to update centroids. Default: 5.",
                        default=5,
                        type=int)

    parser.add_argument("--alpha",
                        help="Value for Leaky ReLU. Default: 0.2.",
                        type=float,
                        default=0.2)

    parser.add_argument("--gamma",
                        help="Value for clustering loss. Default: 10.",
                        type=float,
                        default=10.0)

    parser.add_argument("--hops",
                        help="Hops for proximity matrix. Default: 2.",
                        type=int,
                        default=2)

    parser.add_argument("--epochs",
                        help="Maximum number of epochs. Default: 500.",
                        type=int,
                        default=500)

    parser.add_argument("--patience",
                        help="Epochs to wait for improvement. Default: 50.",
                        type=int,
                        default=50)

    parser.add_argument("--n-init",
                        help="Initializations for K-Means. Default: 20.",
                        type=int,
                        default=20)

    parser.add_argument("--seed",
                        help="Random seed number. Optional.",
                        type=int)

    parser.add_argument("--stage",
                        help="Stage(s) to run. Default: 'pretrain+train'.",
                        choices=["pretrain+train", "pretrain", "train"],
                        default="pretrain+train")

    parser.add_argument("--weights",
                        default="weights/daegc",
                        help="Directory to save or load pretrained model weights. Optional.")

    parser.add_argument("--device",
                        help="Device to use ('cpu' or 'cuda'). "
                             "Default: 'cuda', if available.",
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(getargs()))
