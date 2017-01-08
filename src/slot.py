import numpy as np

class VariableSlot(object):
    def __init__(self, val, trainable):
        self._val = val
        self._grad = None
        self._trainable = trainable

    def load(self, another):
        assert type(another) is VariableSlot
        assert type(self.val) is type(another.val)
        assert self.val.shape == another.val.shape
        self._val = another.val
        assert type(self.grad) is type(another.grad)
        if type(self.grad) is np.ndarray:
            assert self.grad.shape == another.grad.shape
        self._grad = another.grad
        self._trainable = another._trainable

    @property
    def val(self):
        return self._val

    @property
    def grad(self):
        return self._grad
    
    def assign(self, new_val):
        self._val = new_val
    
    def lookup(self, pos):
        return np.take(self._val, pos, axis = 0)

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
        self._val = np.zeros(shape)
        self._trainable = False
        self._alpha = momentum
        self._grad = None
    
    def apply_update(self, new_val):
        self._val *= self._alpha
        self._val += self._alpha * new_val