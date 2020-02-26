# TCEC network sampling

Python implementation of the TCEC sampling algorithm described in:
* [1] N Ruggeri and C De Bacco, _Sampling on Networks: Estimating Eigenvector Centrality on Incomplete Networks_, COMPLEX NETWORKS 2019, SCI 881, pp. 1–12, 2020 (DOI will soon appear online here )

This is a new sampling method to estimate eigenvector centrality on incomplete networks. 
The sampling algorithm is theoretically grounded by results derived from spectral approximation theory.   

The paper can be found here:
* [Published version](https://doi.org/10.1007/978-3-030-36687-2_8) (soon to appear online...).
* [Preprint version](https://arxiv.org/abs/1908.00388).  

If you use this code please cite [1].

Copyright (c) 2019 [Nicolò Ruggeri](https://www.is.mpg.de/person/nruggeri) and [Caterina De Bacco](http://cdebacco.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Complete example
Example of how to run the sampler on a synthetic direct graph generated using `networkx`.  
Alternatively, if your full network is contained in a file, e.g. edgelist format stored in `adjacency.dat`, please first read it as input and transform it into a `networkx` Graph object as in the commented line below. 
This example, along with others, is contained in the file `example.py`.  

```python
import networkx as nx
import numpy as np 
from tcec_sampling import TcecSampler

'''
Input or generate network
'''
#G = nx.read_edgelist('adjacency.dat', create_using=nx.DiGraph()) # input full (giant) network from file named 'adjacency.dat'
G = nx.scale_free_graph(1000)   # generate a directed scale-free graph of 1000 nodes

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
sampler.sample(first_node, directed, sample_size, successors, predecessors=predecessors)

# the sampled subgraph is stored as a sampler attribute, and is a networkx object
subgraph = sampler.subG
print(f'The sampled subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges')
```

## Basic functionality 

### Sampling an undirected network
All the functionalities are provided by the TcecSampler class. To perform exploration one just needs to 
provide a `successors` function that, given a node, returns all its neighbours. 
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
sampler.sample(first_node, sample_size, directed, successors, predecessors=predecessors)
```
 
## Remarks
### Strategies for speeding up the algorithm
The structure of the sampler has been designed to allow sampling out-of-memory graphs. For this reason we need to 
pass the function `successors` (and eventually `predecessors`) as input.

In many scenarios, exploring a node could be the major cost in the sampling procedure. Imagine for example having to
download and scrape an html page, or performing expensive database queries. The choice of how to handle this 
cost is left to the final user. For instance, one should store in memory the calls to these functions in order to avoid performing costly operations repeatedly. As this could be memory heavy, alternatively, one could save the outputs in files that are then re-opened "at need", this could also be a major time saver. 

### Defining nodes
As the sampler relies on the networkx implementation of networks, any valid node in networkx can be used as a node in 
the function. This means that any hashable (therefore, immutable) object can be intended as a node. 
However, the `successors` and `predecessors` functions must be able to accept as input any node object in the network.  


## Other options and output
The TcecSampler.sample method has many options. For a full overview type `help(TcecSampler().sample)`. 
Some of the notable ones are:

- **count_type**: either `'nodes'` or `'edges'`, defaults to `'nodes'`. How to count the sample size.
- **weight_feature**: as the TCEC algorithm works on weighted non negative graphs, one may wish to use a non binary 
edge representation. weight_feature is the key that is used to retrieve the edge value from the adjacency dictionary 
returned by the functions `successors` and `predecessors`. Notice that if passed as input, the argument weight_feature
must be present as key of every adjacency dictionary returned from the two functions.
- **leaderboard_size**: size of the leaderboard kept by the algorithm, as from reference paper [1].
- **neigh_eval_frac**: float in the range (0, 1]. The randomization level `p` in the reference paper. It is the random 
fraction of new neighbours explored at every sampling step.
- **max_time**: float, in seconds. Maximum sampling time before stopping, even if the required sampled size has not been 
reached.
- **save_every_n**: int. Number of sampled nodes or edges (according to count_type) between intermediate savings of the 
sampled subgraph in pickle format. In case this parameter is passed, one should also pass the argument `saving_path`, specifying the path to the saved graph.
- **saving_path**: path. Output file for the sampled network.
- **verbose**: bool. If True, print intermediate messages about the status of the sampling procedure.  


All these arguments are saved as attributes of the TcecSampler instance after the call of `.sample method`. In addition, 
the `.subG` attribute stores the sampled graph as a networkx object.
