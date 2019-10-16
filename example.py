import networkx as nx
import numpy as np

from tcec_sampling.tcec_sampling import tcec_sampling


# Here we propose an example for the usage of our publicly available implementation of the TCEC sampling algorithm
# https://arxiv.org/abs/1908.00388
# We show some simple examples of how to well design the inputs arguments for the provided function.

# Some remarks: the function is intended for use when a graph cannot entirely fit in memory. For this reason the only
# requirement for running the sampling using our function is to give as input two helper functions that, given a node as
# input, return the neighbours of the node itself.
# While for this examples we actually HAVE the graph in memory, the functions successors and predecessors evaluate the
# neighbours in a lazy way, i.e. only when called.
# Notice that our implementation leaves any memoization choice to the end user: if exploration of the graph is expensive
# and explored nodes need to be saved, this has to be handled inside the arguments successors and predecessors.

# We loosely follow the structure of any networkx adjacency dictionary. This means that nodes can be represented by any
# hashable object, as they are keys of an adjacency dictionary, and edges can contain information represented as
# dictionary values. See help(tcec_sampling) for further information.

######################################################
############# directed graph example #################
######################################################

# for directed graphs the sampling algorithm requires two functions: the successors function, returning outgoing
# connections of a node, and the predecessors function, returning the incoming connections.

N_nodes = 5000
print('Generating synthetic graph...')
G = nx.scale_free_graph(N_nodes)   # generating a directed graph
assert G.is_directed()
print('Total graph number of nodes and edges:', G.number_of_nodes(), G.number_of_edges())
successors = lambda node: dict(G[node])
predecessors = lambda node: dict({pred: G.adj[pred][node] for pred in G.predecessors(node)})

N_nodes = 1000
print('\nSampling with minimum requirement on nodes. Number of nodes to be sampled:', N_nodes)
subG_n = tcec_sampling(first_node=np.random.choice(list(G.nodes())), directed=G.is_directed(),
                       predecessors=predecessors, successors=successors,sample_size=N_nodes)
print('Sampled graph. Nodes and edges:', subG_n.number_of_nodes(), subG_n.number_of_edges())


######################################################
############ undirected graph example ################
######################################################

# In the case of an undirected graph the function just needs a successors function, as incoming and outgoing edges are
# the same
N_nodes = 5000
m = 20
print('Generating synthetic graph...')
G = nx.barabasi_albert_graph(N_nodes, m)    # generating an undirected graph
assert not G.is_directed()
print('Total graph number of nodes and edges:', G.number_of_nodes(), G.number_of_edges())
successors = lambda node: dict(G[node])

N_nodes = 1000
print('\nSampling with minimum requirement on nodes. Number of nodes to be sampled:', N_nodes)
subG_n = tcec_sampling(first_node=np.random.choice(list(G.nodes())), directed=G.is_directed(),
                       successors=successors, sample_size=N_nodes)
print('Sampled graph. Nodes and edges:', subG_n.number_of_nodes(), subG_n.number_of_edges())














