from variable import Variable
from utils import extract

"""
A slot is defined to be a dict with pairs (key : value) being 
(variable : {'val' : value, 'grad' : gradient})
each module have a slot on the server.
"""

class Slot(object):
    def __init__(self, module_name, *args, **kwargs):
        self._name = module_name
        trainable, kwargs = extract('trainable', True, **kwargs)
        self._init_vars_and_hyper_pars(trainable, *args, **kwargs)

    def __call__(self, var_name):
        return self._dicts[var_name]

    def _init_vars_and_hyper_pars(
            self, trainable, *args, **kwargs):
        self._hyper_pars = (args, kwargs)
        self._dicts = dict()
    
    @property
    def hyper_pars(self):
        return self._hyper_pars

    @property
    def name(self):
        return self._name
    
    @property
    def var_names(self):
        return self._dicts.keys()

class UniSlot(Slot):
    def _init_vars_and_hyper_pars(
            self, trainable, val, *args, **kwargs):
        self._hyper_pars = (args, kwargs)
        self._dicts = {
            self.name: Variable(
                val, self.name, trainable)
        }

    def __call__(self, *args, **kwargs):
        return self._dicts[self.name]

_slot_class_factory = dict({
    'dot': UniSlot,
    'bias': UniSlot,
    'conv': UniSlot,
})

def slot_class_factory(name):
    return _slot_class_factory.get(name, Slot)