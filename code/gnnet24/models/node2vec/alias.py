import networkx as nx
import numpy as np
from scipy.special import softmax


class AliasSampler:
    """
    A class for sampling from a non-uniform discrete distribution using the alias method.

    The alias method is a linear, fast, and memory-efficient way to draw samples from a discrete distribution.
    It is particularly useful when the number of outcomes is large and the probabilities are not uniform.

    Example to draw N=1000 samples from a random probability vector of K=5 elements:
        >>> probs = np.random.dirichlet(np.ones(5), 1).ravel()
        >>> sampler = AliasSampler(probs)
        >>> draws = [sampler() for _ in range(1000)]

    See [reference](https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/) for details.
    """
    def __init__(self, probs: list) -> tuple:
        """
        Initialize the AliasSampler with a list of probabilities.

        :param probs: A list of probabilities for each outcome.
        """
        self.J, self.q = alias_setup(probs)

    def __call__(self):
        """
        Draw a sample from the non-uniform discrete distribution.

        Returns:
            int: The index of the sampled outcome.
        """
        return alias_draw(self.J, self.q)


def alias_setup(probs: list) -> tuple:
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger  = []

    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J: list, q: list) -> int:
    """
    Draw sample from non-uniform discrete distribution.
    Returns the index of the drawn outcome.

    :param J: A list of indices for the larger outcomes.
    :param q: A list of probabilities for the larger outcomes.
    """
    K = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(np.random.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def alias_setup_nodes(G: nx.Graph, edge_weight: dict = {}) -> dict:
    """
    Compute utility lists for non-uniform sampling from graph nodes.

    :param G: Graph to generate embeddings.
    :param edge_weight: Dictionary with edge weights.
    """
    return {
        node:
            alias_setup(
                normalize([
                    edge_weight.get((node, n), 1)
                    for n in sorted(G.neighbors(node))
                ])
            )
        for node in G.nodes(data=False)
    }


def alias_setup_edges(
    G: nx.Graph,
    p: float = 1.0,
    q: float = 1.0,
    c: float = 1.0,
    r: float = None,
    t: float = 0.0,
    edge_weight: dict = {},
    edge_time: dict = {},
    edge_keys: dict = {},
    reverse: bool = False,
    norm_exp: bool = False,
) -> dict:
    """
    Compute utility lists for non-uniform sampling from graph edges.

    Continuous time intervals are considered as the difference between the
    next and previous edge times. Any interval lower than zero is ignored.

    :param G: Graph to generate embeddings.
    :param p: Higher values promote BFS walks. This parameter is used
        in the original Node2Vec implementation. If set to unity, it
        matches the original DeepWalk implementation. Default: 1.0.
    :param q: Lower values promote DFS walks. This parameter is used
        in the original Node2Vec implementation. If set to unity, it
        matches the original DeepWalk implementation. Default: 1.0.
    :param c: Lower values promote cycles. This parameter is hidden
        in the original Node2Vec implementation. Default: 1.0.
    :param r: Lower values promote graph exploration. Used as an
        alternative to `p`, `q`, `c`. Default: 1.0.
    :param t: Lower values promote temporal exploration.
        Only used when `temporal` is set as True. Default: 0.
    :param edge_weight: Dictionary with edge weights.
    :param edge_time: Dictionary with edge times.
    :param edge_keys: Dictionary with edge keys.
    :param reverse: If True, random walks are computed
        in the reverse direction. Default: False.
    :param norm_exp: If True, transition probabilities are
        normalized using the softmax function. Default: False.
    """
    delta_t, prob = 0, 0
    norm_func = softmax if norm_exp else normalize

    return {
        (u, v, k):
            alias_setup(
                norm_func([
                    prob/np.log(np.e + delta_t)
                    for n in sorted(G.neighbors(v))
                    for k_ in edge_keys.get((v, n), [0])
                    if (
                        not edge_time
                        or (n == u and (delta_t := edge_time.get((v, n, k_), t)) <= 0)
                        or ((prev_t := edge_time.get((v, u, k) if reverse else (u, v, k))) is not None and
                            (next_t := edge_time.get((v, n, k_))) is not None and
                            (delta_t := (next_t - prev_t)) <= 0
                            )
                    )
                    and (prob :=
                         (edge_weight.get((v, n, k_), 1)/(max((1-r), 1) if n == u else r))
                         if r else
                         (edge_weight.get((v, n, k_), 1)/(p if n == u else c if G.has_edge(n, u) else q))
                    ) > 0
                ])
            )
        for u, v, k in [
            (edge[:2][::-1] if reverse else edge[:2]) + (edge[2] if len(edge) == 3 else 0,)
            for edge in (G.edges(keys=True if edge_time else False) if G.is_multigraph() else G.edges())
        ]
    }


def normalize(probs: list) -> list:
    """
    Normalize a list of probabilities.

    :param probs: List of probabilities.
    """
    norm_const = sum(probs)
    return [float(prob)/norm_const for prob in probs]


def softmax(probs: list) -> list:
    """
    Compute the softmax of a list of probabilities.

    :param probs: List of probabilities.
    """
    exp_probs = np.exp([p - np.max(probs) for p in probs])
    return exp_probs/np.sum(exp_probs, axis=0)
