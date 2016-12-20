import numpy as np
from utils import extract

class Optimizer(object):
    def __init__(self, *args, **kwargs):
        minimize, kwargs = extract(
            'minimize', True, **kwargs)
        self._d = -1. * np.float64(minimize)
        self._construct(*args, **kwargs)

    def apply(self, var_slot):
        for var_name in var_slot.names():
            var_slot.set_val(
                var_name, self._cal_val)

class StochasticDescentOptimizer(Optimizer):
    def _construct(self, lr = 1e-4):
        self._lr = lr * self._d

    def _cal_val(self, v, g):
        return v + self._lr * g

"""
Optimizer factory
"""

_optimizer_factory = dict({
    'sgd' : StochasticDescentOptimizer,
})

def optimizer_factory(name, *args, **kwargs):
    assert name in _optimizer_factory, \
    'Optimizer {} not found'.format(name)
    return _optimizer_factory[name](*args, **kwargs)