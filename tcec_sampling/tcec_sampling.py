import math
import os
import time

import networkx as nx
import numpy as np

from .random_walk import random_walk
from .utils import _generic_input_check, _sample_size_is_reached, TopNHeapq, _intermediate_save_if_necessary


class TcecSampler:
    """ Network sampler, implementing the TCEC algorithm https://arxiv.org/abs/1908.00388 """

    def __init__(self):
        self.subG = None
        self._leaderboard = None

        # attributes to give as input to the self.sample method
        self.first_node = None
        self.sample_size = None
        self.directed = None
        self._successors = None
        self._predecessors = None
        self.count_type = None
        self.weight_feature = None
        self.random_walk_init = None
        self.random_walk_type = None
        self.patience = None
        self.leaderboard_size = None
        self.neigh_eval_frac = None
        self.alpha = None
        self.neighs_type = None
        self.max_time = None
        self.save_every_n = None
        self.saving_path = None
        self.verbose = None

    def sample(self, first_node, sample_size, directed, successors, predecessors=None, count_type='nodes',
               weight_feature=None, random_walk_init=0.5, random_walk_type='rw', patience=math.inf,
               leaderboard_size=100, neigh_eval_frac=0.1, alpha=None, neighs_type='incoming', max_time=math.inf,
               save_every_n=None, saving_path=None, verbose=False):
        """ Perform sampling on given graph using the tcec algorithm. All input parameters are saved as TcecSampler
        attributes.

        :param first_node: the node from where to start the search. Compliant with networkx, a node can be any hashable
              object contained in the graph under sampling
        :param sample_size: int, the minimum number of nodes or edges (see count_type) of the final sampled subgraph
        :param directed: bool, if the sampled graph (and therefore, the returned subgraph) is directed
        :param successors: a function of type
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
              is left to the final user, which can decide to memoize successors and predecessors functions realizations, at
              an increased memory cost but avoiding repeated calls.
        :param predecessors: like successors, but the adjacency dictionary must contain as keys all the nodes j that are
              contained in edges like j-->i. predecessors defaults to None, but an error is raised if it is not provided
              when directed=True
        :param count_type: one in ['nodes', 'edges']. If 'nodes', then sample_size is computed as the number of nodes
              visited. If 'edges' the same is done counting the number of edges.
        :param weight_feature: the weight feature that defines the adjacency matrix. Must be present for all edges of G, with
              numeric type and non negative
        :param random_walk_init: the fraction of the required final graph to be explored at the beginning via random walk. If
              int, count the number of nodes or edges. If float 0<sample_size<1, the fraction of nodes or edges of the final
              sample
        :param random_walk_type: the type of random walk used for exploration if random_walk_init > 0
        :param patience: patience of initial random walk
        :param leaderboard_size: the maximum size allowed for the leaderboard
        :param neigh_eval_frac: float, 0 < neigh_frac_eval <=1. The fraction of neighbours to choose at random for criterion
              evaluation (randomization parameter p in reference paper)
        :param alpha: float in [0, 1], parameter used in the theoretical criterion computation. If None, 1 is used for
              undirected graphs and 0.5 for directed ones
        :param neighs_type: one in ['incoming', 'outgoing'], if to define neighbours like incoming or outgoing connections
        :param max_time: float, time limit (in seconds) before stopping, regardless of the size reached, and returning the
              sample. Defaults to math.inf, i.e. no time stopping
        :param save_every_n: int. If passed as input, the number of nodes or edges (according to count_type) to sample
              between automatic saving of sampled graph. If None, no intermediate saving is performed.
        :param saving_path: used only if save_every_n is not None. path to save intermediate sampled graphs.
              If None, save in current working directory
        :param verbose: bool, if to include intermediate messages about the sampling procedure
        :return: networkx.Graph if directed == False, else networkx.DiGraph. Contains the sampled network
        """

        _generic_input_check(directed, count_type, predecessors, save_every_n)

        # store sampling parameters in class attributes
        self.first_node = first_node
        self.sample_size = sample_size
        self.directed = directed
        self._successors = successors
        self._predecessors = predecessors
        self.count_type = count_type
        self.weight_feature = weight_feature
        self.random_walk_type = random_walk_type
        self.patience = patience
        self.leaderboard_size = leaderboard_size
        self.neigh_eval_frac = neigh_eval_frac
        self.neighs_type = neighs_type
        self.max_time = max_time
        self.save_every_n = save_every_n
        self.verbose = verbose

        if saving_path is None and save_every_n is not None:
            self.saving_path = os.path.join(os.getcwd(), 'tcec_sampled_subgraph.pkl')

        # if random_walk_init is a fraction, find the expected sample size from random walk
        if 0 < random_walk_init < 1:
            self.random_walk_init = np.ceil(random_walk_init * self.sample_size)
        else:
            self.random_walk_init = random_walk_init
        if alpha is None:
            self.alpha = 0.5 if self.directed else 0.

        # start sampling procedure
        t0 = time.time()

        # perform random walk initialization if required
        current_node = self.first_node
        if self.random_walk_init > 1:
            self.subG = random_walk(self.first_node, self.random_walk_init, self.directed, self._successors,
                                    self._predecessors, walk_type=self.random_walk_type, count_type=self.count_type,
                                    weight_feature=self.weight_feature, patience=self.patience, verbose=self.verbose)
            border = {neigh for node in self.subG for neigh in
                      self._neighbourhood(node)
                      if neigh not in self.subG}
        else:
            self.subG = nx.DiGraph() if self.directed else nx.Graph()
            self.subG.add_node(self.first_node)
            border = set(self._neighbourhood(current_node).keys())

        # add custom_weight to every node in sample.
        # It is the sum of weights of connections coming from border nodes into a sampled node
        border = set(np.random.choice(list(border), int(len(border) * self.neigh_eval_frac), replace=False))
        for node in self.subG:
            self.subG.nodes[node]['in_deg_weight'] = sum(self._adj_val(neigh, node)
                                                    for neigh in (self._predecessors(node) if self.directed else self._successors(node))
                                                    if neigh in border)
        self._leaderboard = TopNHeapq(n=self.leaderboard_size,
                                      data=[(node,
                                       self._theoretical_criterion(node))
                                      for node in border])

        # start influence increment sampling
        while not self._sample_size_is_reached() and time.time() - t0 < self.max_time:
            if len(self._leaderboard) == 0:
                if self.verbose:
                    print(
                        f'Found empty leaderboard, sample size {self.subG.number_of_nodes() if self.count_type == "nodes" else self.subG.number_of_edges()} {self.count_type}. '
                        f'Recomputing leaderboard on new random neighbours.')
                while len(self._leaderboard) == 0:
                    current_nodes = np.random.choice(list(self.subG.nodes),
                                                     max(int(self.subG.number_of_nodes() * self.neigh_eval_frac), 1)
                                                     )
                    neighs = {node
                              for current_node in current_nodes
                              for node in self._neighbourhood(current_node)
                              if node not in self.subG}
                    self._leaderboard = TopNHeapq(n=self.leaderboard_size,
                                                  data=[(node, self._theoretical_criterion(node))
                                                  for node in neighs])
            # select node and add to the sampled graph
            selected_node = self._leaderboard.pop_max()[0]
            for node in self._successors(selected_node):
                if node in self.subG:
                    self.subG.nodes[node]['in_deg_weight'] -= self._adj_val(selected_node, node) ** 2
            self.subG.add_node(selected_node)
            self.subG.nodes[selected_node]['in_deg_weight'] = 0
            self.subG.add_edges_from((selected_node, neigh, edge_attr)
                                     for neigh, edge_attr in self._successors(selected_node).items() if neigh in self.subG)
            if self.directed:
                self.subG.add_edges_from((in_neigh, selected_node, edge_attr)
                                         for in_neigh, edge_attr in self._predecessors(selected_node).items() if in_neigh in self.subG)

            # update the leaderboard with newly discovered neighbours
            neighs = [neigh for neigh in self._neighbourhood(selected_node) if neigh not in self.subG]
            if len(neighs) > 0:
                neighs = np.random.choice(neighs, max(int(len(neighs) * self.neigh_eval_frac), 1), replace=False)
            # update in_deg_weight and border info
            for neigh in neighs:
                self.subG.nodes[selected_node]['in_deg_weight'] += self._adj_val(neigh, selected_node) ** 2
                if neigh not in border:
                    border.add(neigh)
                    for node in self._successors(neigh):
                        if node in self.subG and node != selected_node:
                            self.subG.nodes[node]['in_deg_weight'] += self._adj_val(neigh, node) ** 2
            # update leaderboard
            for neigh in neighs:
                infl_neigh = self._theoretical_criterion(neigh)
                self._leaderboard.add(neigh, infl_neigh)

            self._intermediate_save_if_necessary()

    def _neighbourhood(self, node):
        return (
            self._successors(node) if not self.directed or self.neighs_type == 'outgoing'
            else self._predecessors(node)
        )

    def _adj_val(self, u, v):
        """
        Numerical value of the edge from u to v. If no edge is present, return 0. Otherwise return the value
        of the attribute weight_feature attached to the edge. If weight_feature is None, return 1 for existing edges.
        """
        succ = self._successors(u)
        if v in succ and self.weight_feature is not None:
            return succ[v][self.weight_feature]
        if v in succ:
            return 1
        return 0

    def _theoretical_criterion(self, node):
        """
        Compute theoretical criterion of node goodness on a given node. The functions successors and predecessors, as well
        as the objects weight_feature and alpha,  are passed as inputs from the main function tcec_sampling.
        subG is the graph sampled up to a given moment.
        """
        b1_T_U_norm = sum(
            self._adj_val(node, neigh) ** 2 * self.subG.nodes[neigh]['in_deg_weight']
            for neigh in self._successors(node) if neigh in self.subG
        )

        if self.subG.is_directed():
            # sum of connections from node to sample
            b1_norm = sum(
                self._adj_val(node, neigh) ** 2 for neigh in self._successors(node) if neigh in self.subG
            )
            # sum of connections from outside the sample to the node
            b3_norm = sum(
                self._adj_val(neigh, node) ** 2 for neigh in self._predecessors(node) if
                neigh not in self.subG
            )

            return b1_norm + (1 - self.alpha) * (b1_T_U_norm - b3_norm)
        else:
            # sum of connections from node to sample minus connections from outside sample to the node.
            # Same as above, but compacting the operation (b1_norm - b3_norm) in the undirected case
            b1_norm_minus_b3_norm = sum(
                (
                    self._adj_val(node, neigh) ** 2 if neigh in self.subG
                    else -(1 - self.alpha) * self._adj_val(neigh, node) ** 2
                )
                for neigh in self._successors(node) if neigh in self.subG
            )
            return b1_norm_minus_b3_norm + (1 - self.alpha) * b1_T_U_norm

    def _sample_size_is_reached(self):
        return _sample_size_is_reached(self.subG, self.sample_size, self.count_type)

    def _intermediate_save_if_necessary(self):
        _intermediate_save_if_necessary(self.subG, self.count_type, self.save_every_n, self.saving_path)





