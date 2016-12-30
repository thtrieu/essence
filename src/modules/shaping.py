from module import Module
from src.utils import nxshape
import numpy as np

class reshape(Module):
    def __init__(self, server, x_shape, new_shape):
        self._x_shape = (x_shape)
        self._out_shape = (new_shape)

    def forward(self, x):
        return x.reshape(
            nxshape(x, self._out_shape))
    
    def backward(self, grad):
        return grad.reshape(
            nxshape(grad, self._x_shape))	

class batch_slice(Module):
    def __init__(self, _, inp_shape, pos_shape, axis, shift):
        self._axis = axis
        self._shift = shift
        self._inp_shape = inp_shape
        self._base = [slice(None)] * len(inp_shape)
        self._out_shape = tuple([
            d for i, d in enumerate(inp_shape) if i != axis])
    
    def forward(self, x, positions):
        batch = range(x.shape[0])
        self._indices = [batch] + self._base
        positions = np.array(positions) + self._shift
        self._indices[self._axis + 1] = positions 
        return x[self._indices]
    
    def backward(self, grad):
        real_shape = nxshape(grad, self._inp_shape)
        base_grad = np.zeros(real_shape)
        base_grad[self._indices] = grad
        return base_grad


class slices(Module):
    def __init__(self, _, inp_shape, item):
        self._item = [slice(None)] + list(item)
        temp = np.zeros(inp_shape)
        self._out_shape = temp[item].shape
        self._inp_shape = inp_shape
    
    def forward(self, x):
        return x[self._item]

    def backward(self, grad):
        g = np.zeros(nxshape(grad, self._inp_shape))
        g[self._item] = grad
        return g