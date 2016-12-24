import numpy as np

class Graph(object):
    def __init__(self):	pass
    def add_portal(self, shape): pass
    def add_node(self, *args, **kwargs): pass
    def forth_traverse(self, target): pass
    def back_traverse(self, target): pass

class Chain(Graph):
    def __init__(self):
        self._list = list()
        self._current_shape = tuple()

    def forth_traverse(self):
        for node in self._list:
            yield node

    def back_traverse(self):
        for node in self._list[::-1]:
            yield node
    
    def set_portal(self, shape):
        self._current_shape = shape

    def add_node(self, node):
        self._list.append(node)
        self._current_shape = node.out_shape

    @property
    def leaf(self):
        return self._list[-1]
    
    @property
    def current_shape(self):
        return self._current_shape