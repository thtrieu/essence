from optimizer import optimizer_factory
from initializer import initializer_class_factory
from utils import extract

class VariableSlot(object):
    def __init__(self, module_name, *args, **kwargs):
        self._name = module_name
        self._trainable, kwargs = extract('trainable', True, **kwargs)

        initializer_class = initializer_class_factory(module_name)
        initializer = initializer_class(*args, **kwargs)
        self._valgrad = initializer.val_grad_pairs

    @property
    def name(self):
        return self._name

    def names(self):
        return self._valgrad.keys()

    def val(self, name):
        return self._valgrad[name]['val']
    
    def grad(self, name):
        return self._valgrad[name]['grad']
    
    def set_grad(self, name, rule, *args):
        if self._trainable: 
            vg = self._valgrad[name]
            val = rule(*args)
            vg['grad'] = val
    
    def set_val(self, name, rule):
        if self._trainable:
            vg = self._valgrad[name]
            val = rule(vg['val'], vg['grad'])
            vg['val'] = val
    
class VariableBank(object):
    """
    A bank of variables, consists of slots
    each slot contains variable of one module, thus
    each slot can have zero, one or more variables.
    """
    
    def __init__(self):
        self._var_slots = list()

    def issue(self, name, *args, **kwargs):
        var_slot = VariableSlot(
            name, *args, **kwargs)
        self._var_slots.append(var_slot)
        return var_slot
    
    def set_optimizer(self, name, *args, **kwargs):
        optimizer = optimizer_factory(
            name, *args, **kwargs)
        self._optimizer = optimizer

    def apply_optimizer(self):
        for var_slot in self._var_slots[::-1]:
            self._optimizer.apply(var_slot)
    
    def save(file_name):
        pass
            
    def load(file_name):
        pass