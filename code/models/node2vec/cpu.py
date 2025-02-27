import random
from typing import Optional

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from .alias import (
    alias_draw,
    alias_setup_edges,
    alias_setup_nodes,
)


class Node2Vec():
    """
    Generates embeddings from a graph using the Node2Vec algorithm.

    This implementation is CPU-bound and relies on the Word2Vec model, as
    implemented in the Gensim library. Alias sampling is used to efficiently
    generate random samples from a discrete probability distribution.

    For more information, please check the
    [documentation](https://radimrehurek.com/gensim/models/word2vec.html).

    :param embedding_dim: Dimension of the embeddings.
    :param walks_per_node: Number of walks per node.
    :param walk_length: Length of each walk.
    :param context_size: The actual context window size considered for
        positive samples. This parameter increases the effective sampling
        rate by reusing samples across different source nodes.
    :param epochs: Number of epochs. Default: 1.
    :param lr: Learning rate. Default: 0.025.
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
    :param num_negative_samples: Number of negative samples to use.
    :param num_workers: Number of parallel workers. Default: 1.
    :param temporal: Whether to employ time-respecting walks.
        An edge-level attribute "time" is required. Default: False.
    :param norm_exp: Whether to employ softmax to normalize transition
        probabilities. Default: False.
    :param seed: Random seed number for predictable randomness.
        Note that `num_workers` > 1 will influence the results.
    """
    def __init__(
        self,
        embedding_dim: int,
        walks_per_node: int,
        walk_length: int,
        context_size: int,
        epochs: int = 1,
        lr: float = 0.025,
        p: float = 1.0,
        q: float = 1.0,
        c: float = 1.0,
        r: Optional[float] = None,
        t: float = 0.0,
        num_negative_samples: int = 1,
        num_workers: int = 1,
        norm_exp: bool = False,
        temporal: bool = False,
        seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.context_size = context_size
        self.epochs = epochs
        self.lr = lr
        self.p = p
        self.q = q
        self.c = c
        self.r = r
        self.t = t
        self.num_negative_samples = num_negative_samples
        self.num_workers = num_workers
        self.norm_exp = norm_exp
        self.temporal = temporal
        self.seed = seed

    def __call__(self, G: nx.Graph, nodes: Optional[list] = None) -> np.ndarray:
        """
        Learn embeddings by walk sampling.
        """
        self.word2vec = Word2Vec(
            self.walk(G, nodes=nodes),
            vector_size=self.embedding_dim,
            window=self.context_size,
            workers=self.num_workers,
            epochs=self.epochs,
            negative=self.num_negative_samples,
            min_count=0,
            sg=1,
            alpha=self.lr,
            seed=self.seed
        )
        return self

    def walk(self, G: nx.Graph, nodes: Optional[list] = None) -> list:
        """
        Generate random walks from a graph.
        """
        walks = []
        nodes = nodes or list(G.nodes())

        # Get edge weights if available.
        edge_weight = {}
        for edge, weight in nx.get_edge_attributes(G, "weight").items():
            edge_weight[edge if G.is_multigraph() else (edge[0], edge[1], 0)] = weight

        # Get edge times if available.
        edge_time = {}
        if self.temporal:
            for edge, time in nx.get_edge_attributes(G, "time").items():
                edge_time[edge if G.is_multigraph() else (edge[0], edge[1], 0)] = time

        # Get edge keys if multigraph.
        edge_keys = {}
        if G.is_multigraph():
            edge_keys = {edge: [] for edge in G.edges(keys=False)}
            for u, v, k in G.edges(keys=True):
                edge_keys[(u, v)].append(k)

        # Precompute the transition probabilities for nodes.
        alias_nodes = alias_setup_nodes(G, edge_weight)

        # Precompute the transition probabilities for edges.
        alias_edges = alias_setup_edges(G, p=self.p, q=self.q, c=self.c, r=self.r, t=self.t,
                                        edge_weight=edge_weight, edge_time=edge_time, edge_keys=edge_keys,
                                        norm_exp=self.norm_exp)

        # If the graph is undirected, the transition probabilities should be symmetric.
        if not G.is_directed():
            alias_edges.update(alias_setup_edges(G, p=self.p, q=self.q, c=self.c, r=self.r, t=self.t,
                                                 edge_weight=edge_weight, edge_time=edge_time, edge_keys=edge_keys,
                                                 norm_exp=self.norm_exp, reverse=True))

        for i in range(self.walks_per_node):
            random.shuffle(nodes)

            # Simulate random walks starting from each node.
            for j, node in enumerate(nodes):
                print(f"({i+1}/{self.walks_per_node}) Walk sampling node {j+1}/{G.order()}...", end="\r")
                walk = [node]

                # Continue sampling neighbors until walk length is reached...
                while len(walk) < self.walk_length:
                    if (
                        not self.temporal
                        and (neighbors := sorted(G.neighbors(walk[-1])))
                    ):
                        walk.append(
                            neighbors[
                                alias_draw(*(
                                    alias_nodes[walk[-1]]
                                    if len(walk) == 1
                                    else alias_edges[walk[-2], walk[-1], 0])
                                )
                            ]
                        )
                    elif (
                        self.temporal
                        and (edges := list(G.edges(walk[-1], keys=True) if G.is_multigraph() else G.edges(walk[-1])))
                    ):
                        walk.append(
                            (edge := edges[alias_draw(*(
                                alias_nodes[walk[-1]]
                                if len(walk) == 1
                                else alias_edges[
                                    edge[0],
                                    edge[1],
                                    edge[2] if G.is_multigraph() else 0
                                ]
                            ))])[1]
                        )
                    else:
                        break
                walks.append(walk)

        return [list(map(str, walk)) for walk in walks]
