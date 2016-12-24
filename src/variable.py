import numpy as np

class Variable(object):
    def __init__(self, val, name, trainable):
        if val.dtype == np.float64:
            val = val.astype(np.float32)
        self._val = val
        self._name = name
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
            if 0:# tuple(self._grad.shape) == (32,):
                print '\n{}: {}\n'.format(self._name, self._grad.shape)
                print self._grad#[:,:,1,2]
                #exit()
    
    def apply_grad(self, apply_rule):
        if self._trainable:
            self._val = apply_rule(self.val, self.grad)
            if 0: #tuple(self._val) == (32,):
                print '\n{}: {}\n'.format(self._name, self._val.shape)
                print self._val