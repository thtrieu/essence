from .module import Module
from src.utils import nxshape
import numpy as np

class concat(Module):
    def __init__(self, server, *args):
        args = list(args)
        axis = args[-1]
        shapes = args[:-1]
        new_size = sum(x[axis] for x in shapes)
        shapes = [list(shp) for shp in shapes]
        for shp in shapes: shp[axis] = None
        
        assert shapes[1:] == shapes[:-1], \
        'All other dim must be equal when concatenate'
        self._axis = axis + 1
        
        new_shp = shapes[0]
        new_shp[axis] = new_size
        new_shp = tuple(new_shp)
        self._out_shape = new_shp

    def forward(self, *args):
        inputs = list(args)
        self._sizes = [
            x.shape[self._axis] \
            for x in inputs]
        return np.concatenate(
            inputs, self._axis)
    
    def backward(self, grad):
        returns = list()
        rank = len(grad.shape)
        slices = [slice(None)] * rank

        offset = int(0)
        for size in self._sizes:
            slices[self._axis] = \
                slice(offset, offset + size)
            returns.append(grad[slices])
            offset = offset + size
        return returns

class transpose(Module):
    def __init__(self, server, x_shape, trans):
        self._out_shape = np.array(x_shape)[trans]
        self._trans = np.array([-1] + trans) + 1

    def forward(self, x):
        return x.transpose(self._trans)
    
    def backward(self, grad): # not tested
        return grad.transpose(
            np.argsort(self._trans))

class reshape(Module):
    def __init__(self, server, x_shape, new_shape, 
                 over_batch = False):
        self._over_batch = over_batch
        self._out_shape = new_shape

    def forward(self, x):
        new_shape = self._out_shape
        if not self._over_batch:
            new_shape = nxshape(x, new_shape)
        self._x_shape = x.shape
        return x.reshape(new_shape)
    
    def backward(self, grad):
        return grad.reshape(self._x_shape)

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

class dynamic_slice(Module):
    def __init__(self, server, xshape, start, end, axis):
        self._axis = axis
        self._out_shape = xshape
    
    def forward(self, x, start, end):
        rank = len(x.shape)
        slices = [slice(None)] * rank
        slices[self._axis + 1] = slice(start, end)
        slices = tuple(slices)
        self._xshape = x.shape
        self._slices = slices
        result = x[slices]
        return result
    
    def backward(self, grad):
        g = np.zeros(self._xshape)
        g[self._slices] = grad
        return g

class slices(Module):
    def __init__(self, _, inp_shape, item):
        self._item = [slice(None)] + list(item)
        temp = np.zeros(inp_shape)
        self._out_shape = temp[item].shape
        self._inp_shape = inp_shape
    
    def forward(self, x):
        t = x[self._item]
        return x[self._item]

    def backward(self, grad):
        g = np.zeros(nxshape(grad, self._inp_shape))
        g[self._item] = grad
        return g