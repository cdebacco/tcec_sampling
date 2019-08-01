import math

import networkx as nx
import numpy as np

from random_walk import random_walk
from utils import _generic_input_check, _sample_size_is_reached, TopNHeapq


def theoretical_criterion_sampling(first_node, sample_size, directed, successors, predecessors=None, count_type='nodes',
                                   weight_feature=None, random_walk_init=0.5, random_walk_type='rw', patience=math.inf,
                                   leaderboard_size=100, neigh_eval_frac=0.1, alpha=None, neighs_type='incoming',verbose=False):
    """
    Search with iterative evaluation of best criterion value, as from reference paper.
    Initialize search with a random walk of size random_walk_init.
    INPUT:
        - first_node: the node from where to start the search. Compliant with networkx, a node can be any hashable
          object.
        - sample_size: int, the minimum number of nodes or edges (see count_type) of the final sampled subgraph
        - directed: bool, if the sampled graph (and therefore, the returned subgraph) is directed
        - successors: a function of type
          f(node) --> adj_dict
          that, given a node i as input, returns an adjacency dictionary specified as follows.
          The keys are all the nodes j pointed from edges i-->j. The values are (eventually empty) dictionaries
          containing values attached to the edge i-->j.
          Therefore adjacency dictionaries look like:
          adj_dict = {
            1: {'weight': 0.3, 'count': 2, ...},
            2: {'weight': 1.2},         # for example the edge i-->2 has attached the attribute 'weight' with value 1.2
            3: dict(),
            ...
          }
          Notice that a node can be any hashable object, but they need to be uniquely identifiable.
          IMPORTANT NOTE: the function successors (as well as predecessors) is called very often. According to the cost
          of the call, one may prefer to store results in memory instead of calling on already seen nodes. This choice
          is left to the final user, which can decide to memoize successors and predecessors functions, at an increased
          memory cost but avoiding repeated calls.
        - predecessors: like successors, but the adjacency dictionary must contain as keys all the nodes j that are
          contained in edges like j-->i. predecessors defaults to None, but an error is raised if it is not provided
          when directed=True
        - count_type: one in ['nodes', 'edges']. If 'nodes', then sample_size is computed as the number of nodes
          visited. If 'edges' the same is done counting the number of edges.
        - weight_feature: the weight feature that defines the adjacency matrix. Must be present for all edges of G, with
          numeric type and non negative
        - random_walk_init: the fraction of the required final graph to be explored at the beginning via random walk. If
          int, count the number of nodes or edges. If float 0<sample_size<1, the fraction of nodes or edges of the final
          sample
        - random_walk_type: the type of random walk used for exploration if random_walk_init > 0
        - patience: patience of initial random walk
        - leaderbpard_size: the maximum size allowed for the leaderboard
        - neigh_eval_frac: float, 0 < neigh_frac_eval <=1. The fraction of neighbours to choose at random for criterion
          evaluation (randomization parameter p in reference paper)
        - alpha: float in [0, 1], parameter used in the theoretical criterion computation. If None, 1 is used for
          undirected graphs and 0.5 for directed ones
        - neighs_type: one in ['incoming', 'outgoing'], if to define neighbours like incoming or outgoing connections
    """
    # check correctness of the inputs
    _generic_input_check(directed, count_type, predecessors)

    # if random_walk_init is a fraction, find the expected sample size from random walk
    if 0 < random_walk_init < 1:
        random_walk_init = np.ceil(random_walk_init*sample_size)
    if alpha is None:
        alpha = 0.5 if directed else 1

    # perform random walk initialization if required
    current_node = first_node
    if random_walk_init > 1:
        subG = random_walk(first_node, random_walk_init, directed, successors, predecessors, walk_type=random_walk_type,
                           count_type=count_type, weight_feature=weight_feature, patience=patience,verbose=verbose)
        border = {neigh for node in subG for neigh in _neighbourhood(node, successors, predecessors, directed, neighs_type)
                  if neigh not in subG}
    else:
        subG = nx.DiGraph() if directed else nx.Graph()
        subG.add_node(first_node)
        border = set(_neighbourhood(current_node, successors, predecessors, directed, neighs_type).keys())

    # add custom_weight to every node in sample.
    # It is the sum of weights of connections coming from border nodes into a sampled node
    border = set(np.random.choice(list(border), int(len(border) * neigh_eval_frac), replace=False))
    for node in subG:
        subG.nodes[node]['in_deg_weight'] = sum(_adj_val(neigh, node, successors, weight_feature)
                                                for neigh in (predecessors(node) if directed else successors(node))
                                                if neigh in border)
    leaderboard = TopNHeapq(n=leaderboard_size,
                            data=[(node, _theoretical_criterion(node, successors, predecessors, subG, weight_feature, alpha))
                                  for node in border])

    # start influence increment sampling
    while not _sample_size_is_reached(subG, sample_size, count_type):
        if len(leaderboard) == 0:
            while len(leaderboard) == 0:
                current_nodes = np.random.choice(list(subG.nodes),
                                                 max(int(subG.number_of_nodes() * neigh_eval_frac), 1)
                                                 )
                neighs = {node
                          for current_node in current_nodes
                          for node in _neighbourhood(current_node, successors, predecessors, directed, neighs_type)
                          if node not in subG}
                leaderboard = TopNHeapq(n=leaderboard_size,
                                        data=[(node, _theoretical_criterion(node, successors, predecessors, subG, weight_feature, alpha))
                                              for node in neighs])
        # select node and add to the sampled graph
        selected_node = leaderboard.pop_max()[0]
        for node in successors(selected_node):
            if node in subG:
                subG.nodes[node]['in_deg_weight'] -= _adj_val(selected_node, node, successors, weight_feature)**2
        subG.add_node(selected_node)
        subG.nodes[selected_node]['in_deg_weight'] = 0
        subG.add_edges_from((selected_node, neigh, edge_attr)
                            for neigh, edge_attr in successors(selected_node).items() if neigh in subG)
        if directed:
            subG.add_edges_from((in_neigh, selected_node, edge_attr)
                                for in_neigh, edge_attr in predecessors(selected_node).items() if in_neigh in subG)

        # update the leaderboard with newly discovered neighbours
        neighs = [neigh for neigh in _neighbourhood(selected_node, successors, predecessors, directed, neighs_type)
                  if neigh not in subG]
        if len(neighs) > 0:
            neighs = np.random.choice(neighs, max(int(len(neighs) * neigh_eval_frac), 1), replace=False)
        # update in_deg_weight and border info
        for neigh in neighs:
            subG.nodes[selected_node]['in_deg_weight'] += _adj_val(neigh, selected_node, successors, weight_feature)**2
            if neigh not in border:
                border.add(neigh)
                for node in successors(neigh):
                    if node in subG and node != selected_node:
                        subG.nodes[node]['in_deg_weight'] += _adj_val(neigh, node, successors, weight_feature)**2
        # update leaderboard
        for neigh in neighs:
            infl_neigh = _theoretical_criterion(neigh, successors, predecessors, subG, weight_feature, alpha)
            leaderboard.add(neigh, infl_neigh)
    return subG


def _neighbourhood(node, successors, predecessors, directed, neighs_type):
    if not directed or neighs_type == 'outgoing':
        return successors(node)
    if neighs_type == 'incoming':
        return predecessors(node)


def _theoretical_criterion(node, successors, predecessors, subG, weight_feature, alpha):
    b1_T_U_norm = sum(
        _adj_val(node, neigh, successors, weight_feature)**2 * subG.nodes[neigh]['in_deg_weight']
        for neigh in successors(node) if neigh in subG
    )

    if subG.is_directed():
        # sum of connections from node to sample
        b1_norm = sum(
            _adj_val(node, neigh, successors, weight_feature)**2 for neigh in successors(node) if neigh in subG
        )
        # sum of connections from outside the sample to the node
        b3_norm = sum(
            _adj_val(neigh, node, successors, weight_feature)**2 for neigh in predecessors(node) if neigh not in subG
        )

        return b1_norm + (1 - alpha) * (b1_T_U_norm - b3_norm)
    else:
        # sum of connections from node to sample minus connections from outside sample to the node.
        # Same as above, but compacting the operation (b1_norm - b3_norm) in the undirected case
        b1_norm_minus_b3_norm = sum(
            (
                _adj_val(node, neigh, successors, weight_feature)**2 if neigh in subG
                else -(1 - alpha) * _adj_val(neigh, node, successors, weight_feature)**2
            )
            for neigh in successors(node) if neigh in subG
        )
        return b1_norm_minus_b3_norm + (1 - alpha) * b1_T_U_norm


def _adj_val(u, v, successors, weight_feature):
    """
    Numerical value of the edge from u to v. If no edge is present, return 0. Otherwise return the value
    of the attribute weight_feature attached to the edge. If weight_feature is None, return 1 for existing edges.
    """
    succ = successors(u)
    if v in succ and weight_feature is not None:
        return succ[v][weight_feature]
    if v in succ:
        return 1
    return 0




