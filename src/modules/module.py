import numpy as np

class Module(object):
    def __init__(self, server, *shapes_and_params): pass        
    def forward(self, x): pass 
    def backward(self, grad): pass
    
    @property
    def out_shape(self):
        return tuple(self._out_shape)

class portal(Module):
    def __init__(self, server, shape):
        self._inp_shape = tuple(shape)
        self._out_shape = tuple(shape)

    def forward(self, x):
        return x

    def backward(self, grad):
        return None