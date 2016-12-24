import numpy as np

class Module(object):
    
    # dynamic batch shared by 
    # all Module instances
    _batch = None

    def __init__(self, slot, shape, 
                *args, **kwargs):
        self._var = slot
        self._inp_shape = shape[1:]
        self._out_shape = None
        self._setup(*args, **kwargs)

        if self._out_shape is None:
            self._out_shape = self._inp_shape
        print '{:<20}:{:>20} -> {:<20}'.format(
            slot.name, shape, self.out_shape)

    def _setup(self, *args, **kwargs): pass
    def forward(self, x): pass 
    def backward(self, grad): pass

    @classmethod
    def _set_batch(cls, batch):
        cls._batch = batch
    
    @classmethod
    def _batch_times(cls, shape):
        shp = [cls._batch] + list(shape)
        return tuple(shp)

    @property
    def out_shape(self):
        if self._out_shape is None:
            self._out_shape = self._inp_shape
        return Module._batch_times(self._out_shape)

    @property
    def inp_shape(self):
        return Module._batch_times(self._inp_shape)

class portal(Module):
    def _setup(self, shape):
        self._inp_shape = tuple(shape)
        self._out_shape = tuple(shape)

    def forward(self, x):
        Module._set_batch(x.shape[0])
        return x

    def backward(self, grad):
        pass # no gradient for input.
        # except, when you're dreaming