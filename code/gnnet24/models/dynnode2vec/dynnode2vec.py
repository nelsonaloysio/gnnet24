import logging as log
import os.path as osp

import networkx as nx

from .graph import get_delta_nodes
from ..node2vec.cpu import Node2Vec


def dynnode2vec(
    graphs: list,
    embedding_dim: int,
    walks_per_node: int,
    walk_length: int,
    context_size: int,
    epochs: int = 1,
    lr: float = 0.025,
    p: float = 1.0,
    q: float = 1.0,
    num_negative_samples: int = 1,
    num_workers: int = 1,
    norm_exp: bool = False,
    seed: int = None,
    merge_snapshots: bool = False,
    output_snapshots: bool = False,
    output: str = "embeddings.emb",
) -> None:
    """
    Generates embeddings from a list of graph snapshots using the DynNode2Vec algorithm.

    Reference paper:
    [dynnode2vec: Scalable Dynamic Network Embedding](https://arxiv.org/abs/1812.02356).
    """
    output = osp.splitext(output)[0]

    model = Node2Vec(
        embedding_dim=embedding_dim,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        context_size=context_size,
        epochs=epochs,
        lr=lr,
        p=p,
        q=q,
        num_negative_samples=num_negative_samples,
        num_workers=num_workers,
        norm_exp=norm_exp,
        temporal=True,
        seed=seed,
    )

    for i, current_graph in enumerate(graphs):
        log.info(f"Processing graph {i+1}/{len(graphs)}...")

        if i == 0:
            model(current_graph)

        else:
            if merge_snapshots:
                current_graph = nx.compose(current_graph, graphs[i])

            walks = model.walk(current_graph, nodes=get_delta_nodes(current_graph, previous_graph))
            model.word2vec.build_vocab(walks, update=True)
            model.word2vec.train(walks, total_examples=model.word2vec.corpus_count, epochs=model.epochs)

        if output_snapshots:
            model.word2vec.wv.save_word2vec_format(f"{output}_t={i}.emb")
        if i+1 == len(graphs):
            model.word2vec.wv.save_word2vec_format(f"{output}.emb")

        previous_graph = current_graph
