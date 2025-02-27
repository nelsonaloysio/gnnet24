import json
import logging as log
from itertools import product


def load_params(filepath: str) -> dict:
    """ Load parameters from JSON file. """
    with open(filepath, "r", encoding="utf8") as f:
        return {f"--{k}: {v}" for k, v in json.load(f).items()}


def grid_search(*args, **kwargs) -> list:
    """ Build grid search permutations. """
    grid = {}

    # Wrap key-value pairs into dictionary.
    if args:
        key = None
        for i, arg in enumerate(args):
            # --key=arg
            if arg.startswith("--") and "=" in arg:
                key, value = arg.split("=")
                grid[key] = grid.get(key, []) + [value]
                key = None
            # --key [<arg>, ...]
            elif i+1 < len(args) and arg.startswith("--") and not args[i+1].startswith("--"):
                key, value = args[i], args[i+1]
                grid[key] = grid.get(key, []) + [value]
            # [..., <arg>, ...]
            elif key and not arg.startswith("--") and not args[i-1].startswith("--"):
                grid[key][-1] = f"{grid[key][-1]} {arg}"
            # --arg [or] arg
            elif not args[i-1].startswith("--"):
                grid[""] = grid.get("", []) + [arg]

    # Unwrap key-value pairs from dictionary.
    if kwargs:
        for k, v in kwargs.items():
            grid[k] = grid.get(k, []) + (v if isinstance(v, list) else [v])

    # Build permutations for grid search.
    grid = sorted([
        [_ for _ in sum(zip(grid.keys(), values), ()) if _ != ""]
        for values in set(product(*grid.values()))
    ])

    if len(grid) > 1:
        log.info(f"Grid search exploration: {len(grid)} permutations.")
        list(log.info(f"#{i}: {' '.join(g)}") for i, g in enumerate(grid))

    return grid
