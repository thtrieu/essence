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
    
    def lookup(self, pos):
        return np.take(self._val, pos, axis = 0)
    
    @property
    def grad(self):
        return self._grad

    def set_grad(self, grad):
        if self._trainable:
            if self._grad is None:
                self._grad = grad
            else:
                self._grad += grad
    
    def apply_embedding_grad(self, apply_rule):
        grad, pos = self._grad
        
        emb_size = grad.shape[-1]
        grad = grad.reshape(-1, emb_size)
        pos = pos.flatten()
        sum_over = pos[:, None] == pos
        grad_sum = sum_over.dot(grad)

        val_pos = self._val[pos]
        new_val = apply_rule(val_pos, grad)
        self._val[pos] = new_val
        
    def apply_grad(self, apply_rule):
        if self._trainable and self._grad is not None:
            if type(self._grad) is tuple: 
                self.apply_embedding_grad(apply_rule)
            else:
                self._val = apply_rule(self.val, self.grad)
            self._grad = None
    
class MovingVariableSlot(VariableSlot):
    def __init__(self, shape, momentum):
        self._val = np.zeros(shape, dtype = np.float32)
        self._trainable = False
        self._alpha = momentum
        self._grad = None
    
    def apply_update(self, new_val):
        self._val *= self._alpha
        self._val += self._alpha * new_val