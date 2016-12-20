import numpy as np

class Optimizer(object):
    def __init__(self, minimize = True,
                 *args, **kwargs):
        self._d = -1. * np.float64(minimize)
        self._construct(*args, **kwargs)

    def apply(var_slot):
        for var_name in var_slot.names():
            var_slot.set_val(var_name,
                self._cal_val(
                    var_slot.val(var_name),
                    var_slot.grad(var_name),
            ))

class StochasticDescentOptimizer(Optimizer):
    def _construct(self, learning_rate = 1e-4):
        self.lr = 1e-4

    def _cal_val(v, g):
        return v + self._d * g

"""
Optimizer factory
"""

_optimizer_factory = dict({
    'sgd' : StochasticDescentOptimizer,
})

def optimizer_factory(name, minimize = True, *args, **kwargs):
    assert name in _optimizer_factory, \
    'Optimizer {} not found'.format(name)
    return _optimizer_factory[name]