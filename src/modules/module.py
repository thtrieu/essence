import numpy as np

class Module(object):
    def __init__(self, server, shapes, *params):
        self._var = \
        server.issue_slot(
            type(self), shapes, *params)
        args, kwargs = self._var.hyper_pars
        self._prepare(*(shapes + list(args)), **kwargs)

    def out_shape(self, batch_size = None):
        return tuple([batch_size] + 
            list(self._out_shape))

    def _prepare(self, shape, *args, **kwargs): pass
    def forward(self, x): pass 
    def backward(self, grad): pass


class ChainModule(Module):
    def _prepare(self, shape, *args):
        self._inp_shape = shape[1:]
        self._out_shape = None
        self._setup(*args)
        if self._out_shape is None:
            self._out_shape = self._inp_shape
        print '{:>20} -> {:<20}'.format(
            shape[0], self.out_shape())

    def inp_shape(self, batch_size = None):
        return tuple([batch_size] + 
            list(self._inp_shape))

    def _setup(self, *args, **kwargs): pass


class portal(ChainModule):
    def _setup(self, shape):
        self._inp_shape = tuple(shape)
        self._out_shape = tuple(shape)

    def forward(self, x):
        return x

    def backward(self, grad):
        return None