from numbers import Number
import heapq


def _sample_size_is_reached(subG, sample_size, count_type):
    """ Function for verifying if stopping criterion is satisfied """
    if count_type == 'edges':
        return subG.number_of_edges() >= sample_size
    if count_type == 'nodes':
        return subG.number_of_nodes() >= sample_size


def _generic_input_check(directed, count_type, predecessors):
    if count_type not in ['nodes', 'edges']:
        raise ValueError("count_type input must be equal to 'nodes' or 'edges'")
    if directed and predecessors is None:
        raise ValueError('since the sampled graph is directed, a function predecessors must be provided as input.'
                         'If instead you are sampling from an undirected graph, set directed=False')


# TopNHeapq is needed for keeping the leaderborad of best nodes in the border of the explored graph.
# Used inside influence_increment_sampling. We want to store a heap with elements of type (node_name, influence)
# and keep a max heap. As heapq implements a min heap instead, we build a specific data structure for this

##############################################
##############################################
# COMMENTS: the TopNHeapq is a heap with some additional features. The important ones are:
# - the fact that the maximum size is set, regulated by the method .add, which pops a node before adding a new one if
#   the max size is reached (if the node is already in the heap it modifies its infleucen in place) . The sorting should
#   be preserved in any case by the next call of pop_max, but I didn't notice this detail and need to check carefully
# - there is an extra attribute of the class, which is Node._node_names. It is essentially just a set of the nodes
#   present in the heap, since the heap itself is not hashable. This isn't memory heavy, as the heap should stay small,
#   but allows lookups of cost O(1) instead of O(n) (with n size of the heap). It is more important to have fast lookups
#   as we may need to perform many
###############################################
##############################################

class TopNHeapq:
    """
    max-heap with optional maximum size limit.
    The input data must contain elements of type (node, influence)
    """
    def __init__(self, n, data=[]):
        self.max_size = n
        self._removed_token = '<REMOVED>'

        # initialize the list of nodes
        def node_if_valid(node, influence):
            if node == self._removed_token:
                raise ValueError('no node can have name', self._removed_token,
                                 'which is a protected expression for instances of', self.__class__)
            return Node(node, -influence)

        if len(data) > 0:
            # invert sign of numerical value, as we want to turn min-heap into max-heap
            data = list(sorted([node_if_valid(node, influence) for node, influence in data]))[:self.max_size]
            heapq.heapify(data)

        self.data = data
        self._node_names = {node.node_name for node in data}
        if len(self.data) != len(self._node_names):
            raise ValueError('there are nodes with same name (even if possibly different value). Give unique ' +
                             'identifiers as node names')
        self._removed_count = 0

    def __len__(self):
        return len(self.data) - self._removed_count

    def __iter__(self):
        for node in self.data:
            if not node.node_name == self._removed_token:
                yield (node.node_name, -node.influence)

    def pop_max(self):
        node = heapq.heappop(self.data)
        # pop nodes until a valid one is picked. Since the non valid ones are not included in the nodes_names
        # there is no need to act on Node._node_names
        while node.node_name == self._removed_token:
            self._removed_count -= 1
            node = heapq.heappop(self.data)
        self._node_names.remove(node.node_name)
        return node.node_name, -node.influence

    def add(self, node_name, influence):
        """ To add a node to the heap structure, just pass the node name and its influence value """
        if node_name in self._node_names:
            idx = self.data.index(Node(node_name, -influence))
            self.data[idx].node_name = self._removed_token
            self._removed_count += 1
            heapq.heappush(self.data, Node(node_name, -influence))
        else:
            if len(self) == self.max_size:
                node_out = heapq.heappop(self.data)
                # it is not optimal to pop with heapq.heappop and then push with heapq.heappush.
                # It should be done with heapq.heappushpop, but we need to check for lazy deletes at every heappop
                while node_out.node_name == self._removed_token:
                    self._removed_count -= 1
                    node_out = heapq.heappop(self.data)
                # check if the node to be inserted has bigger influence than the one popped (nodes are inserted with
                # value -influence, as this is a min-heap, therefore check if the new node has -influence lesser than
                # the popped one
                if node_out.influence > -influence:
                    self._node_names.remove(node_out.node_name)
                    heapq.heappush(self.data, Node(node_name, -influence))
                    self._node_names.add(node_name)
                else:
                    # reinserting the node out has cost O(1), as it has just been popped, nut has the advantage of
                    # cleaning self.data from lazy deletes during the popping procedure
                    heapq.heappush(self.data, node_out)
            else:
                heapq.heappush(self.data, Node(node_name, -influence))
                self._node_names.add(node_name)

    def remove(self, node_name):
        if node_name not in self._node_names:
            raise ValueError('The input', node_name, 'is not contained in the elements of the heap')
        idx = self.data.index(Node(node_name))
        self.data[idx] = Node(self._removed_token, None)
        self._node_names.remove(node_name)
        self._removed_count += 1


class Node:
    def __init__(self, node_name, influence=None):
        self.node_name = node_name
        self.influence = influence

    def __repr__(self):
        return '({}, {})'.format(self.node_name, -self.influence)

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.influence < other.influence
        raise NotImplementedError('cannot compare type ', type(self), 'with type', type(other))

    def __gt__(self, other):
        if isinstance(other, Node):
            return self.influence > other.influence
        raise NotImplementedError('cannot compare type ', type(self), 'with type', type(other))

    # the __eq__ and __hash__methods have to be coherent. Since two nods are defined equal if they have the same
    # node_name attribute (regardless of the score attribute), the __hash__ method will invoke the hashing on
    # Node.node_name
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_name == other.node_name
        raise NotImplementedError('cannot compare type ', type(self), 'with type', type(other))

    def __hash__(self):
        return self.node_name.__hash__()

    def __mul__(self, other):
        if isinstance(other, Number):
            return Node(self.node_name, self.influence*other)
        raise NotImplementedError('cannot multiply an object of type ', type(self), 'with one of type', type(other))

    __rmul__ = __mul__

    def inplace_scalar_mul(self, other):
        if isinstance(other, Number):
            self.influence *= other
        else:
            raise NotImplementedError('cannot multiply an object of type ', type(self), 'with one of type', type(other))
