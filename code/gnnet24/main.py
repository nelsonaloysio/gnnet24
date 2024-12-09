#!/usr/bin/env python3

from functools import partial
import logging as log
import os
import os.path as osp
import sys
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Tuple

PATH = Path((osp.abspath(osp.dirname(__file__))))
if PATH not in sys.path:
    sys.path.append(str(PATH))

MODELS = sorted(os.listdir(osp.join(PATH, "models")))[1:]  # Ignore __init__.py
DATASETS = sorted(os.listdir(osp.join(PATH.parent.parent, "data")))

from .dataset import GNNetDataset
from .utils.grid_search import grid_search, load_params
from .utils.transform import transform


def getargs(argv: list = sys.argv[1:]) -> Tuple[Namespace, list]:
    """ Get arguments from command line. """
    parser = ArgumentParser(#add_help=False,
                            allow_abbrev=False,
                            description="Command line interface wrapper.")

    parser.add_argument("model",
                        choices=sorted(MODELS),
                        help=f"Model to run. Choices: {MODELS}.")

    parser.add_argument("name",
                        choices=[data.lower() for data in DATASETS],
                        help="Dataset name to load.")

    parser.add_argument("--features",
                        dest="pretrained_features",
                        metavar="FILE_PATH",
                        help="Path to file with node features. Overrides default choice.")

    parser.add_argument("--features-pretrained", "--pretrained",
                        choices=["node2vec", "dynnode2vec", "tnodeembed"],
                        default=None,
                        dest="pretrained_features",
                        help="Choice of pretrained node features to load.")

    parser.add_argument("--no-features",
                        action="store_const",
                        const=False,
                        dest="pretrained_features",
                        help="Use one-hot encoding (identity matrix) as node features.")

    parser.add_argument("--static",
                        action="store_false",
                        dest="temporal",
                        help="Disregard temporal data by setting edge times to 0).")

    # parser.add_argument("--directed",
    #                     action="store_true",
    #                     help="Remove edges added by PyG when converting a directed graph.")

    parser.add_argument("--discretized",
                        action="store_true",
                        help="Whether to sort and discretize edge times to integers.")

    parser.add_argument("--normalized",
                        action="store_true",
                        help="Normalize features for unit mean and zero variance.")

    parser.add_argument("--split",
                        choices=["transductive", "inductive", "temporal"],
                        default=None,
                        help="Learning setting. If unset, consider the whole graph.")

    parser.add_argument("--train-ratio",
                        type=float,
                        help="Proportion of training edges. Optional.")

    parser.add_argument("--val-ratio",
                        type=float,
                        help="Proportion of validation edges. Optional.")

    parser.add_argument("--level-ratio",
                        choices=["node", "edge"],
                        default="node",
                        help="Whether train_ratio and val_ratio are node-level or edge-level. "
                             "Choices: ('node', 'edge'). Default is 'node'.")

    parser.add_argument("--log-file", "--log",
                        default=None,
                        metavar="LOG_FILE",
                        help="File to save log. Optional.")

    parser.add_argument("--log-level",
                        choices=["debug", "info", "warning", "error", "critical", "notset"],
                        default="info",
                        help="Logging level. Default: 'info'.")

    parser.add_argument("--log-format",
                        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        metavar="LOG_FORMAT",
                        help="Log messages format. Optional.")

    parser.add_argument("--no-log",
                        action="store_const",
                        const="notset",
                        dest="log_level",
                        help="Disable logging.")

    parser.add_argument("--params",
                        default={},
                        metavar="JSON_FILE",
                        type=load_params,
                        help="Load model parameters from JSON file. Optional.")

    return parser.parse_known_args(argv)


def main(argv: list = sys.argv[1:]) -> None:
    """ Main function. """
    args, unwargs = getargs(argv)

    assert args.temporal or args.split != "temporal",\
        "Temporal split requires `temporal=True`."
    assert args.discretized or args.split != "temporal",\
        "Temporal split requires `discretized=True`."

    if args.log_level != "notset":
        log.basicConfig(
            level=args.log_level.upper(),
            format=args.log_format,
            handlers=(
                [log.StreamHandler(stream=sys.stderr)] +
                ([log.FileHandler(args.log_file, mode="w")] if args.log_file else [])
            )
        )

    model = import_module(f"{PATH.name}.models.{args.model}.main")
    log.info("Imported '%s' model.", args.model)

    root = PATH.parent.parent.joinpath("data")

    transform_partial = partial(
        transform,
        pretrained_features=(
            osp.join(root, args.name, "raw", args.pretrained_features or "", "x")
            if args.pretrained_features else None
        ),
        temporal=args.temporal,
        normalized=args.normalized,
        discretized=args.discretized,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        level_ratio=args.level_ratio,
    )

    data = GNNetDataset(root=root, name=args.name, transform=transform_partial)[0]
    log.info("Loaded '%s' dataset: %s.", args.name, data)

    assert hasattr(data, "x") and data.x is not None,\
        "Dataset does not have node features. Use one of: "\
        "--no-features, --pretrained NAME, or --features PATH."

    grid = grid_search(*unwargs, **args.params)
    for argp in grid:
        model_args = model.getargs(argp)
        model_args.__dict__.pop("data")
        log.info("Arguments: %s.", vars(model_args))
        model_args.data = data
        model.main(args=model_args)


if __name__ == "__main__":
    sys.exit(main())
