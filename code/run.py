#!/usr/bin/env python3

import json
import logging as log
import os.path as osp
import sys
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Tuple

import torch

PATH = Path(osp.dirname(__file__))
if PATH not in sys.path:
    sys.path.append(PATH)

from models.utils.data import load_dataset
from models.utils.grid_search import grid_search

MODELS = (
    "attri2vec",
    "daegc",
    "dynnode2vec",
    "kmeans",
    "leiden",
    "node2vec",
    "spectral",
    "tgc",
    "tnodeembed",
)

FEATURES = (
    "node2vec",
    "dynnode2vec",
    "tnodeembed",
)


def getargs(argv: list) -> Tuple[Namespace, list]:
    """ Get arguments from command line. """
    parser = ArgumentParser(add_help=False,
                            allow_abbrev=False,
                            description="Command line interface wrapper.")

    parser.add_argument("model",
                        help=f"Model to run. Choices: {MODELS}.",
                        choices=sorted(MODELS))

    parser.add_argument("--data",
                        dest="name",
                        help="Processed dataset path or name.",
                        required=True)

    parser.add_argument("--features", "--x",
                        choices=FEATURES,
                        default=None,
                        dest="x",
                        help="Pretrained features to load. If unset, original"
                             "dataset features are loaded, if available.")

    parser.add_argument("--split",
                        choices=["transductive", "inductive"],
                        default=None,
                        help="Learning setting. If unset, "
                             "training/validation/test masks are not loaded.")

    parser.add_argument("--static",
                        action="store_false",
                        dest="temporal",
                        help="Disregard temporal data (edge times set to 0).")

    parser.add_argument("--directed",
                        action="store_true",
                        help="Remove edges added by PyG when converting a "
                             "directed graph.")

    parser.add_argument("--normalize",
                        action="store_true",
                        help="Normalize node features for unit mean and zero "
                             "variance.")

    parser.add_argument("--log-level",
                        choices=["debug", "info", "warning", "error",
                                 "critical", "notset"],
                        default="INFO",
                        help="Logging level. Default: 'info'.",
                        type=lambda x: getattr(log, x.upper()))

    parser.add_argument("--log-file",
                        default=None,
                        help="File to save log. Optional.",
                        metavar="LOG_FILE",
                        type=Path)

    parser.add_argument("--params",
                        default={},
                        help="JSON file with additional parameters for grid "
                             "search. Optional.",
                        metavar="JSON_FILE",
                        type=load_params)

    return parser.parse_known_args(argv)


def load_params(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf8") as f:
        return {f"--{k}: {v}" for k, v in json.load(f).items()}


def main(argv: list = sys.argv[1:]) -> None:
    """ Main function. """
    args, unwargs = getargs(argv)

    if args.log_file:
        log.root.handlers = []
        log.root.handlers.append(log.FileHandler(args.log_file, mode="w"))

    if args.log_level:
        log.basicConfig(
            level=args.log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=args.log_file
        )

    model = import_module(f"models.{args.model}.main")

    if type(args.name) == str:
        root = f"{PATH.parent}/data"

        data = load_dataset(
            root=root,
            name=args.name,
            features=(
                "x.npy"
                if args.x is None and osp.isfile(f"{root}/{args.name}/x.npy")
                else f"x_{args.x}.npy"
            ),
        )
    else:
        data = torch.load(args.name)

    log.info(f"Loaded '{args.name}' dataset: {data}")

    for params in grid_search(*unwargs, **args.params):
        model_params = model.getargs([p for p in params if p != ""])
        model_params.__dict__.pop("data")

        log.info(f"Running {args.model} with parameters: {model_params}")
        model_params.data = data
        model.main(args=model_params)


if __name__ == "__main__":
    sys.exit(main())
