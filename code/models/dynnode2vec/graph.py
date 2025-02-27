from itertools import chain
from typing import Any, Optional

import networkx as nx

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj


def get_neighbors(edge_index: Adj, num_nodes: Optional[int] = None):
    """
    Get neighbors of each node in the graph based on edge index.
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    neighbors = {i: set() for i in range(num_nodes)}
    list(neighbors[u].add(v) for u, v in edge_index.t().numpy())

    return {i: list(neighbors[i]) for i in range(num_nodes)}


def get_delta_nodes(current_graph: nx.Graph, previous_graph: nx.Graph) -> set[Any]:
    """
    Original implementation by DynNode2Vec@pedugnat, here extended to multigraphs:
        https://github.com/pedugnat/dynnode2vec/blob/master/dynnode2vec/utils.py

    Find nodes in the current graph which have been modified, i.e., they
    have either been added or at least one of their edge have been updated.
    This is the subset of nodes for which we will generate new random walks.

    We compute the output of equation (1) of the paper, i.e.,

        ∆V_t = V_add U {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add U E_del)}.

    We make the assumption about V_add that we only care about nodes that are
    connected to at least one other node, i.e. nodes that have at least one edge.

    This assumption yields that:

        V_add ⊆ {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add U E_del)},

    which we can use to avoid computing V_add.
    """
    current_edges = set(e for e in current_graph.edges)
    previous_edges = set(e for e in previous_graph.edges)

    # Edges that were either added or removed between current and previous graphs.
    delta_edges = current_edges ^ previous_edges

    # Nodes in the current graph for which edges have been updated.
    # = {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add ∪ E_del)}
    nodes_with_modified_edges = set(chain(*delta_edges))

    # Delta nodes are new nodes (V_add) and current nodes which edges have changed.
    # Since we only care about nodes that have at least one edge, we assume that:
    # V_add ⊆ {v_i ∈ V_t | ∃e_i = (v_i, v_j) ∈ (E_add ∪ E_del)}.
    delta_nodes = list(current_graph.nodes & nodes_with_modified_edges)

    return delta_nodes
