import networkx as nx
import numpy as np

from theoretical_criterion_search import theoretical_criterion_sampling


def tester_theoretical_criterion_sampling(verbose=False):

    # test on erdos renyi undirected graph
    print('\n\nStarting on undirected graph...')
    N_nodes = 5000
    p = 0.05
    G = nx.erdos_renyi_graph(N_nodes, p)
    print('Total graph number of nodes and edges:', G.number_of_nodes(), G.number_of_edges())
    successors = lambda node: dict(G[node])

    N_nodes_s = 50
    print('\nSampling with minimum requirment on nodes. Minimum number:', N_nodes_s)
    subG_n = theoretical_criterion_sampling(first_node=np.random.choice(list(G.nodes())), directed=G.is_directed(),
                                            successors=successors, sample_size=N_nodes_s,
                                            count_type='nodes',  patience=1000, neigh_eval_frac=0.1,verbose=verbose)
    print('Sampled graph. Nodes and edges:', subG_n.number_of_nodes(), subG_n.number_of_edges())

    N_edges_s = 30
    print('\nSampling with minimum requirment on edges. Minimum number:', N_edges_s)
    subG_e = theoretical_criterion_sampling(first_node=np.random.choice(list(G.nodes())), directed=G.is_directed(),
                                            successors=successors, sample_size=N_nodes_s,
                                            count_type='edges', patience=1000, neigh_eval_frac=0.1,verbose=verbose)
    print('Sampled graph. Nodes and edges:', subG_e.number_of_nodes(), subG_e.number_of_edges())

    # test on preferential attachment directed graph
    print('\n\nStarting on directed graph...')
    N_nodes = 5000
    G = nx.scale_free_graph(N_nodes)
    print('Total graph number of nodes and edges:', G.number_of_nodes(), G.number_of_edges())
    successors = lambda node: dict(G[node])
    predecessors = lambda node: dict({pred: G.adj[pred][node] for pred in G.predecessors(node)})

    N_nodes_s = 50
    print('\nSampling with minimum requirment on nodes. Minimum number:', N_nodes_s)
    subG_n = theoretical_criterion_sampling(first_node=np.random.choice(list(G.nodes())), directed=G.is_directed(),
                                            patience=1000, predecessors=predecessors, successors=successors,
                                            sample_size=N_nodes_s, count_type='nodes', neigh_eval_frac=0.1,verbose=verbose)
    print('Sampled graph. Nodes and edges:', subG_n.number_of_nodes(), subG_n.number_of_edges())

    N_edges_s = 10
    print('\nSampling with minimum requirment on edges. Minimum number:', N_edges_s)
    subG_e = theoretical_criterion_sampling(first_node=np.random.choice(list(G.nodes())), directed=G.is_directed(),
                                            patience=1000, predecessors=predecessors, successors=successors,
                                            sample_size=N_nodes_s, count_type='edges', neigh_eval_frac=0.1,verbose=verbose)
    print('Sampled graph. Nodes and edges:', subG_e.number_of_nodes(), subG_e.number_of_edges())


tester_theoretical_criterion_sampling(verbose=True)













