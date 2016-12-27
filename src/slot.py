from variable import Variable
from utils import extract
import numpy as np

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
            'uni': Variable(val, trainable)
        }

    def __call__(self, *args, **kwargs):
        return self._dicts['uni']

class BatchnormSlot(Slot):
    def _init_vars_and_hyper_pars(
        self, trainable, shapes, gamma, mv_mean, mv_var, *args, **kwargs):
        self._hyper_pars = (args, kwargs)
        self._dicts = {
            'gamma': Variable(gamma, trainable),
            'mean': Variable(mv_mean, False),
            'var': Variable(mv_var, False),
        }

class LSTMSlot(Slot):
    def _init_vars_and_hyper_pars(
        self, trainable, shapes, hidden_size, *args, **kwargs):

        args = [hidden_size] + list(args)
        self._hyper_pars = (args, kwargs)

        _, input_size, time_step = shapes[0]
        h, i = hidden_size, input_size
        w_shape = (h, h); i_shape = (i + h, h)
        xavier = np.sqrt(6.)/(np.sqrt(i+2*h))
        
        wi = np.random.uniform(-xavier, xavier, w_shape)
        wo = np.random.uniform(-xavier, xavier, w_shape)
        wf = np.random.uniform(-xavier, xavier, w_shape)
        w  = np.random.uniform(-xavier, xavier, w_shape)
        bi = np.random.ones(h) * .1
        bo = np.random.ones(h) * .1
        bf = np.random.ones(h) * 1.5 # forget bias
        b = np.random.ones(h) * .1 

        self._dicts = {
            'wi': Variable(wi, trainable),
            'wo': Variable(wo, trainable),
            'wf': Variable(wf, trainable),
            'w': Variable(w, trainable),
            'bi': Variable(bi, trainable),
            'bo': Variable(bo, trainable),
            'bf': Variable(bf, trainable),
            'b': Variable(b, trainable)
        }

_slot_class_factory = dict({
    conv: UniSlot,
    matmul: UniSlot,
    add_biases: UniSlot,
    batch_norm: BatchnormSlot
})

def slot_class_factory(module_type):
    return _slot_class_factory.get(module_type, Slot)