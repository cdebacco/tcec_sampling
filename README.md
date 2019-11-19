# TCEC Sampling

Implementing the TCEC sampling algorithm (https://arxiv.org/abs/1908.00388) in Python.

## Basic functionality 

### Sampling an undirected network
All the functionalities are provided by the TcecSampler class. To perform exploration one just needs to 
provide define a `successors` function that, given a node, returns all its neighbours. 
This will be passed as input at sampling time and called by the sampler when the neighbours of
a node are needed by the algorithm.
Notice that by _neighbours_ of a node in an undirected graph we mean all the nodes connected to it.

The code would look something like the following:

```python
from tcec_sampling import TcecSampler

def successors():
    pass   # define function 
    
directed = False   #  we need to know if the sampled network is directed or not
sample_size = 10   # desired final sample size

sampler = TcecSampler()
sampler.sample(first_node, sample_size, directed, successors)
```

As the method `TcecSampler.sample` is based on networkx (https://networkx.github.io/documentation/latest/) 
the function `successors` has to return a structure similar to that of an adjacency dictionary.
In case one just wants to return all the neighbours of a node, the adjacency dictionary provides as keys the new 
neighbours and as values empty dictionaries:

```
successors(node) -> {
    neighbour_1: dict(),  
    neighbour_2: dict(),
    ...,
    neighbour_n: dict()
}
```

To store edge attributes in the sampled graph, just provide them in the adjacency dictionary returned by the function
`successors`:

```
successors(node) -> {
    neighbour_1: {'weight': 2.0},  
    neighbour_2: dict(),
    ...,
    neighbour_n: {'weight': 1, 'collection_date': '02/05/2001'}
}
```

### Sampling a directed network
In the case of a directed network we need to distinguish between _incoming_ and _outgoing_ edges.
Thus, at sampling time, we need to provide two functions, `successors` and `predecessors`. 
`successors` returns an adjacency dictionary, as described above, containing all the outgoing edges of a node
(the _successors_ of a node). `predecessors` returns an adjacency dictionary, with same 
structure as `successors`, but containing _incoming_ edges. In this case the code would look like the following:

```python
from tcec_sampling import TcecSampler

def successors():
    pass

def predecessors():
    pass
    
directed = True   
sample_size = 10   

sampler = TcecSampler()
sampler.sample(first_node, sample_size, directed, successors)
```
 
## Complete example
The code is contained in the file `example.py`. We generate a synthetic graph and sample it using the
TCEC sampling algorithm

```python
import networkx as nx
import numpy as np 
from tcec_sampling import TcecSampler

# generate a directed scale-free graph of 1000 nodes
G = nx.scale_free_graph(1000)   

sampler = TcecSampler()

# parameters for sampling
first_node = np.random.choice(list(G.nodes()))
directed = True
sample_size = 100

successors = lambda node: dict(G[node])
predecessors = lambda node: dict({pred: G.adj[pred][node] for pred in G.predecessors(node)})
print('Example of output from the successors function:', successors(53))
print('Example of output from the predecessors function:', predecessors(10))

# start sampling procedure
sampler.sample(first_node, directed, sample_size, successors, successors)

# the sampled subgraph is stored as a sampler attribute, and is a networkx object
subgraph = sampler.subG
print(f'The sampled subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges')
```

## Remarks
The structure of the sampler has been designed to allow sampling of out-of-memory graphs. For this reason we need to 
pass the function `successors` (and eventually `predecessors`) as input.

In many scenarios, exploring a node could be the major cost in the sampling procedure. The choice of how to handle this 
cost is left to the final user. Notably, one may wish to memoize the calls to these functions not to perform costly 
operations more times. As this could be memory heavy, also saving the outputs in files that are then re-opened at need 
could be a major time saver. 

## Other options and output
The TcecSampler.sample method has many options. For a full overview type `help(TcecSampler().sample)`. 
Some of the notable ones are:

- count_type: either `'nodes'` or `'edges'`, defaults to `'nodes'`. How to count the sample size
- weight_feature: as the TCEC algorithm works on weighted non negative graphs, one may wish to use a non binary 
edge representation. weight_feature is the key that is used to retrieve the edge value from the adjacency dictionary 
returned by the functions `successors` and `predecessors`. Notice that if passed as input, the argument weight_feature
must be present as key of every adjacency dictionary returned from the two functions.
- leaderboard_size: size of the leaderboard kept by the algorithm, as from reference paper
- neigh_eval_frac: float in the range (0, 1]. The randomization level `p` in the reference paper. It is the random 
fraction of new neighbours explored at every sampling step
- max_time: float, in seconds. maximum sampling time before stopping, even if the required sampled size has not been 
reached
- save_every_n: int. Number of sampled nodes or edges (according to count_type) between intermediate savings of the 
sampled subgraph in pickle format. In case it is passed, also the argument saving_path, specifying the path to the saved
 graph, has to be passed.
- verbose: bool. If True, print intermediate messages about the status of the sampling procedure


All these arguments are saved as attributes of the TcecSampler instance after the call of .sample method. In addition, 
the .subG attribute stores the sampled graph as a networkx object.