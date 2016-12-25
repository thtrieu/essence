from variable import Variable
from utils import extract

"""
A slot is defined to be a dict with pairs (key : value) being 
(variable : {'val' : value, 'grad' : gradient})
each module have a slot on the server.
"""

from modules import conv, matmul, add_biases, batch_norm

class Slot(object):
    def __init__(self, *args, **kwargs):
        trainable, kwargs = extract('trainable', True, **kwargs)
        self._init_vars_and_hyper_pars(trainable, *args, **kwargs)

    def __call__(self, var_name):
        return self._dicts[var_name]

    def _init_vars_and_hyper_pars(
            self, trainable, shapes, *args, **kwargs):
        self._hyper_pars = (args, kwargs)
        self._dicts = dict()
    
    @property
    def hyper_pars(self):
        return self._hyper_pars
    
    @property
    def var_names(self):
        return self._dicts.keys()

class UniSlot(Slot):
    def _init_vars_and_hyper_pars(
            self, trainable, shapes, val, *args, **kwargs):
        self._hyper_pars = (args, kwargs)
        self._dicts = {
            'uni': Variable(
                val, 'uni', trainable)
        }

    def __call__(self, *args, **kwargs):
        return self._dicts['uni']

class BatchnormSlot(Slot):
    def _init_vars_and_hyper_pars(
        self, trainable, shapes, gamma, mv_mean, mv_var, *args, **kwargs):
        self._hyper_pars = (args, kwargs)
        self._dicts = {
            'gamma': Variable(gamma, 'gamma', trainable),
            'mean': Variable(gamma, 'mean', False),
            'var': Variable(gamma, 'var', False),
        }

class GRUSlot(Slot):
    def _init_vars_and_hyper_pars(
        self, trainable, shapes, hidden_size, *args, **kwargs):
        self._hyper_pars = (args, kwargs)
        self._dicts = {
            'wr': Variable(None, 'wr', trainable),
            'ur': Variable(None, 'ur', trainable),
            'br': Variable(None)
        }

_slot_class_factory = dict({
    conv: UniSlot,
    matmul: UniSlot,
    add_biases: UniSlot,
    batch_norm: BatchnormSlot
})

def slot_class_factory(module_type):
    return _slot_class_factory.get(module_type, Slot)