import math
import os
import warnings
import random

import networkx as nx
import numpy as np

from .utils import (
    _generic_input_check,
    _sample_size_is_reached,
    _intermediate_save_if_necessary,
)


def random_walk(
    first_node,
    sample_size,
    directed,
    successors,
    predecessors=None,
    walk_type="rw",
    count_type="nodes",
    weight_feature=None,
    patience=math.inf,
    restart_patience=10,
    save_every_n=None,
    saving_path=None,
    verbose=False,
):
    """Sample from the input network G using random walk exploration.

    :param first_node: the node from where to start the search. Compliant with networkx, a node can
          be any hashable object
    :param sample_size: int, the minimum number of nodes or edges (see count_type) of the final
          sampled subgraph
    :param directed: bool, if the sampled graph (and therefore, the returned subgraph) is directed
    :param successors: a function of type
          f(node) --> adj_dict
          that, given a node i as input, returns an adjacency dictionary specified as follows.
          The keys are all the nodes j pointed from edges i-->j. The values are other (eventually
          empty) dictionaries containing values attached to the edge i-->j.
          Adjacency dictionaries look like:
          adj_dict = {
            1: {'weight': 0.3, 'count': 2, ...},
            2: {'weight': 1.2},         # for example the edge i-->2 has attached the attribute 'weight' with value 1.2
            3: dict(),
            ...
          }
          Notice that a node can be any hashable object, but they need to be all different
    :param predecessors: like successors, but the adjacency dictionary must contain as keys all the
          nodes j that are contained in edges like j-->i. predecessors defaults to None, but an error
          is raised if it is not provided when directed=True
    :param walk_type: one of the following
          'rw': standard random walk. At every step, choose a random neighbour.
                Always used when weight_feature!=None
           'mhrw': metropolis-hastings random walk. Correct probabilities using neighbours degree
           'degree_weighted_rw': random walk weighted on the degree of the neighbours. A node is chosen with
                                 probability proportional to its degree
           'degree_greedy': at every step, choose the neighbour with highest degree in a greedy manner
    :param count_type: one in 'nodes', 'edges'. If 'nodes', then sample_size is computed as the
          number of nodes visited. If 'edges' the same is done counting the number of edges
    :param weight_feature: the name of the weight feature to consider when weighting edges during
          the random walk. These values have to be of numeric type and non-negative.
          If None, edges are selected uniformly in the neighbourhood of the node at every step
    :param patience: maximum number of sampling steps where no new nodes are discovered (the walker can keep visiting
           already visited nodes...). If passed, sampling is prematurely interrupted and the current sampled graph is
           returned
    :param restart_patience: maximum number of restart from the same first_node. Valid for directed networks where a
          walker can get stuck in a dangling node
    :param save_every_n: int. If passed as input, the number of nodes or edges (according to count_type) to sample
          between automatic saving of sampled graph. If None, no intermediate saving is performed.
    :param saving_path: used only if save_every_n is not None. path to save intermediate sampled graphs.
          If None, save in current working directory
    :param verbose: bool, if to include intermediate messages about the sampling procedure
    :return: networkx.Graph if directed == False, else networkx.DiGraph. Contains the sampled network
    """

    # check correctness of the inputs
    _generic_input_check(directed, count_type, predecessors, save_every_n)
    if saving_path is None and save_every_n is not None:
        saving_path = os.path.join(os.getcwd(), "sampled_graph_tmp.pkl")

    supported_walks = ["rw", "mhrw", "degree_weighted_rw", "degree_greedy"]
    if walk_type not in supported_walks:
        raise ValueError(
            "The value assigned to walk_type input is not supported. Please choose a value among",
            supported_walks,
        )

    # subG is the sampled and then returned subgraph
    subG = nx.DiGraph() if directed else nx.Graph()
    subG.add_node(first_node)

    # start random walk
    current_node = first_node
    patience_count = 0
    restart_count = 0
    while not _sample_size_is_reached(subG, sample_size, count_type):
        try:
            current_node, current_successors = _random_walk_step(
                current_node,
                walk_type,
                directed,
                successors,
                predecessors,
                weight_feature,
            )
        except ValueError:
            """
            Raised if stuck in a dangling node, i.e. node with no out-going edges.
            In this case we restart from scratch the walk, from the first node again, and erase the sample discovered so far
            """
            restart_count += 1
            if verbose:
                print(f"restart_count for random walk: {restart_count}")

            if restart_count > restart_patience:
                warning_message = (
                    "random walk has not reached the required sample size of {0} {3}, since it has passed the "
                    "maximum number of {1} restart iterations. Stuck too many times in a dangling node. Returning the partial sample collected of "
                    "size {2} {3}".format(
                        sample_size,
                        restart_patience,
                        subG.number_of_nodes()
                        if count_type == "nodes"
                        else subG.number_of_edges(),
                        "nodes" if count_type == "nodes" else "edges",
                    )
                )
                warnings.warn(warning_message)
                return subG

            subG = nx.DiGraph() if directed else nx.Graph()
            subG.add_node(first_node)
            current_node = first_node
            current_successors = successors(current_node)
            patience_count = -1

        if current_node in subG.nodes():
            patience_count += 1
        # check if patience count over or no outgoing nodes or only self loop present (for directed graph)
        if patience_count > patience:
            warning_message = (
                "random walk has not reached the required sample size of {0} {3}, since it has passed the "
                "maximum number of {1} patience iterations. Returning the partial sample collected of "
                "size {2} {3}".format(
                    sample_size,
                    patience,
                    subG.number_of_nodes()
                    if count_type == "nodes"
                    else subG.number_of_edges(),
                    "nodes" if count_type == "nodes" else "edges",
                )
            )
            warnings.warn(warning_message)
            return subG
        subG.add_node(current_node)
        subG.add_edges_from(
            (current_node, other, ebunch)
            for other, ebunch in current_successors.items()
            if other in subG
        )
        if directed:
            subG.add_edges_from(
                (other, current_node, ebunch)
                for other, ebunch in predecessors(current_node).items()
                if other in subG
            )

        _intermediate_save_if_necessary(subG, count_type, save_every_n, saving_path)

    return subG


def _random_walk_step(
    current_node, walk_type, directed, successors, predecessors, weight_feature
):
    """
    Helper function for random_walk, perform a random walk step.
    See random_walk docstring for information. Returns the new visited node and the successors of current_node
    """
    succ = successors(current_node)

    if len(succ) == 0 or len(succ) == 1 and list(succ.keys())[0] == current_node:
        raise ValueError(
            "Trying to perform a random walk step on a node with no other neighbours"
        )

    if weight_feature is not None:
        probs = np.array([succ[x][weight_feature] for x in succ if x != current_node])
        probs /= probs.sum()
        # avoid using np.random.choice directly on succ list, as transforming into numpy array changes the internal type
        # of some objects (e.g. str to numpy.str). This may cause type compatibility errors.
        next_node_idx = np.random.choice(len(probs), size=1, p=probs)[0]
        next_node = [node for node in succ if node != current_node][next_node_idx]
    else:
        # if no weight feature is given, sample among neighbours
        if walk_type == "rw":
            next_node = random.choice([x for x in succ if x != current_node])
        elif walk_type == "mhrw":
            if directed:
                pred = predecessors(current_node)
                probs = np.array(
                    [
                        min(1.0, len(pred) / len(predecessors(neigh)))
                        for neigh in succ
                        if neigh != current_node
                    ]
                )
            else:
                probs = np.array(
                    [
                        min(1.0, len(succ) / len(successors(neigh)))
                        for neigh in succ
                        if neigh != current_node
                    ]
                )
            # sometimes, if the starting node has degree zero, all probabilities are 0. In this case
            # choose a neighbour at random. The convergence of the markov chain is not affected by
            # starting node choice
            if np.allclose(probs, 0):
                probs = np.ones_like(probs)
            # normalize to sum 1
            probs /= sum(probs)
            next_node_idx = np.random.choice(len(probs), size=1, p=probs)[0]
            next_node = [node for node in succ if node != current_node][next_node_idx]
        elif walk_type == "degree_weighted_rw":
            if directed:
                probs = np.array([len(predecessors(node)) for node in succ if node != current_node])
            else:
                probs = np.array([len(successors(node)) for node in succ if node != current_node])
            probs = probs / sum(probs)
            next_node_idx = np.random.choice(len(probs), size=1, p=probs)[0]
            next_node = [node for node in succ if node != current_node][next_node_idx]
        elif walk_type == "degree_greedy":
            if directed:
                next_node = max(succ, key=lambda x: len(predecessors(x)))
            else:
                next_node = max(succ, key=lambda x: len(successors(x)))
    return next_node, succ
