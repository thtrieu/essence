from .modules import module_class_factory
from .utils import parse_for
from .node import Node

class ptr(object):
    pass            
    
class DAG(object):
    def __init__(self):
        self._node_pool = dict()
        # key : val = Ptr : Node

    def register_for_slot(self, name, server, *args):
        # Step 1, extract dependencies from arguments
        dep_parsed = parse_for(ptr, *args)
        dep_list, parameters = dep_parsed

        # Step 2, Get shapes from dependencies
        dep_nodes = list()
        dep_shapes = list()
        for i in range(len(dep_list)):
            node = self._node_pool[dep_list[i]]
            dep_shapes.append(node.out_shape())
            dep_nodes.append(node)

        # Step 3, Create module, wrap by a node, add to pool
        module = module_class_factory(name)(
            server, *(dep_shapes + parameters))
        new_ptr = ptr()
        new_node = Node(module, *dep_nodes)
        self._node_pool[new_ptr] = new_node
        return new_ptr   
    
    def forward_to_leaf(self, fetches, feed):
        fetch = fetches + [self._leaf]
        return self.forward(fetch, feed)
    
    def forward(self, fetches, feed):
        assert type(fetches) is list,\
        'Fetches must be a list'

        forward_id = object()
        vals = list()
        for ptr in feed:
            node = self._node_pool[ptr]
            node.set_feed(feed[ptr])
        for ptr in fetches:
            node = self._node_pool[ptr]
            vals.append(node.forward(forward_id))
        return vals
    
    def set_closure(self, ptr):
        self._leaf = ptr
        leaf_node = self._node_pool[self._leaf]
        leaf_node.inc_counter(ptr)
    
    def backward(self):
        leaf_node = self._node_pool[self._leaf]
        leaf_node.backward(1.0, self._leaf)