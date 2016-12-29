import numpy as np
from utils import extract

class Optimizer(object):
    def __init__(self, lr = 1e-3, *args, **kwargs):
        minimize, kwargs = extract('minimize', True, **kwargs)
        self._lr = lr * (2. * np.float32(minimize) - 1.)
        self._construct(*args, **kwargs)

    def apply(self, var_slot):
        self._current = var_slot
        var_slot.apply_grad(self._rule)
        self._current = None

    def finalize_step(self): pass
    def _construct(*args, **kwargs): pass

class StochasticDescentOptimizer(Optimizer):
    def _construct(self, decay = 1.):
        self._decay = decay

    def _rule(self, v, g):
        return v - self._lr * g
    
    def finalize_step(self):
        self._lr *= self._decay


class AdamOptimizer(Optimizer):
    def _construct(self, p1 = .9, p2 = .999):
        self._p1, self._p2 = p1, p2
        self._moments = dict()
    
    def _rule(self, v, g):
        c = self._current
        m = self._moments
        if c not in m:
            m[c] = dict({'s': 0, 'r': 0, 't': 0})

        s, r, t = m[c]['s'], m[c]['r'], m[c]['t']
        s = s * self._p1 + (1. - self._p1) * g
        r = r * self._p2 + (1. - self._p2) * g * g        
        m[c]['s'], m[c]['r'], m[c]['t'] = s, r, (t + 1)

        s_ = np.divide(s, 1. - np.power(self._p1, t + 1))
        r_ = np.divide(r, 1. - np.power(self._p2, t + 1))
        dv = self._lr * np.divide(s_, np.sqrt(r_) + 1e-8)
        return v - dv
        

"""
Optimizer factory
"""

_optimizer_factory = dict({
    'sgd' : StochasticDescentOptimizer,
    'adam': AdamOptimizer
})

def optimizer_factory(name, *args, **kwargs):
    assert name in _optimizer_factory, \
    'Optimizer {} not found'.format(name)
    return _optimizer_factory[name](*args, **kwargs)