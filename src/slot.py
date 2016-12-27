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

class LSTMSlot(Slot):
    def _init_vars_and_hyper_pars(
        self, trainable, shapes, hidden_size, *args, **kwargs):
        # TODO: let the module init vars itself.
        args = [hidden_size] + list(args)
        self._hyper_pars = (args, kwargs)

        _, input_size, time_step = shapes[0]
        h, i = hidden_size, input_size
        w_shape = (h, h); i_shape = (i, h)
        xavierw = -np.sqrt(6.)/(np.sqrt(2*h))
        xavieru = -np.sqrt(6.)/(np.sqrt(h+i))
        
        wi = np.random.uniform(-xavierw, xavierw, w_shape)
        wo = np.random.uniform(-xavierw, xavierw, w_shape)
        wf = np.random.uniform(-xavierw, xavierw, w_shape)
        w  = np.random.uniform(-xavierw, xavierw, w_shape)
        ui = np.random.uniform(-xavieru, xavieru, i_shape)
        uo = np.random.uniform(-xavieru, xavieru, i_shape)
        uf = np.random.uniform(-xavieru, xavieru, i_shape)
        u  = np.random.uniform(-xavieru, xavieru, i_shape)
        bi = np.random.zeros(h)
        bo = np.random.zeros(h)
        bf = np.random.zeros(h)
        b = np.random.zeros(h)

        self._dicts = {
            'wi': Variable(wi, 'wi', trainable),
            'wo': Variable(wo, 'wo', trainable),
            'wf': Variable(wf, 'wf', trainable),
            'w': Variable(w, 'w', trainable),
            'ui': Variable(ui, 'ui', trainable),
            'uo': Variable(uo, 'uo', trainable),
            'uf': Variable(uf, 'uf', trainable),
            'u': Variable(u, 'u', trainable),
            'bi': Variable(bi, 'bi', trainable),
            'bo': Variable(bo, 'bo', trainable),
            'bf': Variable(bf, 'bf', trainable),
            'b': Variable(b, 'b', trainable)
        }

_slot_class_factory = dict({
    conv: UniSlot,
    matmul: UniSlot,
    add_biases: UniSlot,
    batch_norm: BatchnormSlot
})

def slot_class_factory(module_type):
    return _slot_class_factory.get(module_type, Slot)