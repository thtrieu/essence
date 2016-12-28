import numpy as np

class VariableSlot(object):
    def __init__(self, val, trainable):
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
    
    def set_grad(self, grad):
        if self._trainable:
            self._grad = grad
    
    def apply_grad(self, apply_rule):
        if self._trainable and self._grad is not None:
            self._val = apply_rule(self.val, self.grad)
            self._grad = None
    
class MovingVariableSlot(object):
    def __init__(self, shape, momentum):
        self._val = np.zeros(shape, dtype = np.float32)
        self._alpha = momentum
    
    @property
    def val(self):
        return self._val
    
    def apply_update(self, new_val):
        self._val *= self._alpha
        self._val += self._alpha * new_val
    
    def apply_grad(self, *args):
        pass