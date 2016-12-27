import numpy as np

class Variable(object):
    def __init__(self, val, trainable):
        if val is not None:
            if val.dtype == np.float64:
                val = val.astype(np.float32)
        self._val = val
        self._grad = None
        self._trainable = trainable

    @property
    def val(self):
        return self._val
    
    @property
    def grad(self):
        return self._grad
    
    def set_grad(self, chain_rule, *args):
        if self._trainable:
            self._grad = chain_rule(*args)
    
    def apply_grad(self, apply_rule):
        if self._trainable and self._grad is not None:
            self._val = apply_rule(self.val, self.grad)
            self._grad = None
    
    def apply_update(self, apply_rule, *args):
        self._val = apply_rule(self.val, *args)