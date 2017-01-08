import numpy as np

class Node(object):
    def __init__(self, module, *dep):
        self._depend = list(dep)
        self._module = module
        self._feed = None
        self._cache = None
        self._serve = dict()
        self._count = dict()
        self._grad_cache = 0
        self._forward_id = None
    
    def get_module(self):
        return self._module
    
    def set_feed(self, val):
        self._feed = val

    def out_shape(self):
        return self._module.out_shape
    
    def inc_counter(self, ptr):
        serve = self._serve.get(ptr, 0)
        self._serve[ptr] = serve + 1
        for node in self._depend:
            node.inc_counter(ptr)

    def forward(self, f_id):
        if f_id != self._forward_id:
            self._forward_id = f_id
            val = list()
            if self._feed is None:
                for node in self._depend:
                    val.append(node.forward(f_id))
            elif self._feed is not None:
                val = [self._feed]
                self._feed = None
            self._cache = self._module.forward(*val)
        return self._cache
    
    def backward(self, grad, ptr):
        count = self._count.get(ptr, 0)
        self._count[ptr] = count + 1
        self._grad_cache = self._grad_cache + grad
        if count + 1 == self._serve[ptr]:
            grads = self._module.backward(self._grad_cache)
            self._count[ptr] = self._grad_cache = 0.
            
            if grads is None: return
            if type(grads) is np.ndarray: grads = [grads]
            for g, node in zip(grads, self._depend):
                if g is None: continue
                node.backward(g, ptr)