from module import Module
from src.utils import xavier, guass
import numpy as np
from activations import *

class Recurring(Module):
    """
    Recurring is an internally used module that cache intermediate
    activations in a stack during unrolling of recurrent layers
    Its backward() is expected to be called several times during
    Back Propagation Through Time, either full or truncated.
    """
    def __init__(self, *args, **kwargs):
        self._stack = list()
        self._setup(*args, **kwargs)

    def _setup(*args, **kwargs):
        pass

    def _push(self, *objs):
        objs = list(objs)
        if len(objs) == 1:
            objs = objs[0]
        self._stack.append(objs)
    
    def _pop(self):
        objs = self._stack[-1]
        del self._stack[-1]
        return objs

    def size(self):
        return len(self._stack)

class gate(Recurring):
    """ 
    Gates are special case of Recurring modules
    It starts with a linear transformation
    And end with an activation
    """
    def _setup(self, server, w_shape, 
               bias = None, act_class = sigmoid):
        b_shape = (w_shape[-1],)
        self._act_class = act_class
        w_init = xavier(w_shape)
        if bias is not None:
            b_init = np.ones(b_shape) * bias
        elif bias is None:
            b_init = guass(0.1, 1e-2, b_shape)
        self._w = server.issue_var_slot(w_init, True)
        self._b = server.issue_var_slot(b_init, True)
    
    def forward(self, x):
        linear = x.dot(self._w.val) + self._b.val
        act = self._act_class(None, None)
        self._push(x, act)
        return act.forward(linear)
        
    def backward(self, grad):
        x, act = self._pop()
        linear_grad = act.backward(grad)

        self._b.set_grad(linear_grad.sum(0))
        self._w.set_grad(x.T.dot(linear_grad))
        return linear_grad.dot(self._w.val.T)