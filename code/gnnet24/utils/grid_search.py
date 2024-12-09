import json
from itertools import product


def grid_search(*args, **kwargs) -> list:
    """ Build grid search permutations. """
    grid = {}
    params = []

    # Wrap key-value (arg-param) pairs in dictionary.
    if args:
        key = None
        for i, arg in enumerate(args):
            # --arg=param
            if arg.startswith("--") and "=" in arg:
                key, value = arg.split("=")
                grid[key] = grid.get(key, []) + [value]
                key = None
            # --arg [<param>, ...]
            elif len(args) > i+1 and arg.startswith("--") and not args[i+1].startswith("--"):
                key = arg
            # --arg [..., <param>, ...]
            elif key is not None and not arg.startswith("--"):
                grid[key] = grid.get(key, []) + [arg]
            # ... [arg] or [--arg] (without params)
            else:
                params += [arg]
                key = None

    # Unwrap key-value pairs from dictionary.
    if kwargs:
        for k, v in kwargs.items():
            grid[f"--{k}"] = grid.get(k, []) + (v if isinstance(v, list) else [v])

    # Build permutations for grid search.
    grid = sorted([
        [str(_) for _ in sum(zip(grid.keys(), values), ()) if _ != ""]
        for values in set(product(*grid.values()))
    ])

    return [params + [p for p in perm if p != ""] for perm in grid]


def load_params(filepath: str) -> dict:
    """ Load parameters for grid search from JSON file. """
    with open(filepath, "r", encoding="utf8") as f:
        return {k: v for k, v in json.load(f).items()}
