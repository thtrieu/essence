from modules import module_class_factory
from utils import parse_for
from node import Node

class ptr(object):
    pass            
    
class DAG(object):
    def __init__(self):
        self._var_pool = dict()
        # key : val = Ptr : Node

    def register_for_slot(self, name, server, *args):
        # Step 1, extract dependencies from arguments
        parsed = parse_for(ptr, *args)
        dep_list, parameters = parsed

        # Step 2, extract trainable variables
        variables_slot = \
        server.issue_slot(name, *parameters)
        hyper_pars = variables_slot.hyper_pars
        args, kwargs = hyper_pars

        # Step 3, create module with hyper_pars
        dep_nodes = list()
        shape_list = list()
        for i in range(len(dep_list)):
            node = self._var_pool[dep_list[i]]
            shape_list.append(node.out_shape())
            dep_nodes.append(node)
        if shape_list == list(): shape_list = [(0,)]

        module = module_class_factory(name)(
            variables_slot, *(shape_list+list(args)), **kwargs)
        
        # Step 4, add module to var pool
        new_ptr = ptr()
        self._var_pool[new_ptr] = Node(module, *dep_nodes)
        return new_ptr   
    
    def forward_to_leaf(self, feed):
        fetch = [self._leaf]
        return self.forward(fetch, feed)[0]
    
    def forward(self, fetches, feed):
        assert type(fetches) is list,\
        'Fetches must be a list'
        vals = list()
        forward_id = object()
        for ptr in feed:
            node = self._var_pool[ptr]
            node.set_feed(feed[ptr])
        for ptr in fetches:
            node = self._var_pool[ptr]
            vals.append(node.forward(forward_id))
        return vals
    
    def set_closure(self, ptr):
        self._leaf = ptr
        leaf_node = self._var_pool[self._leaf]
        leaf_node.inc_counter(ptr)
    
    def backward(self):
        leaf_node = self._var_pool[self._leaf]
        leaf_node.backward(1.0, self._leaf)
        
    