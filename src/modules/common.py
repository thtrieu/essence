from .module import Module
from src.utils import binomial, nxshape
import numpy as np

class batch_norm(Module):
    def __init__(self, server, inp_shape,
                gamma_shape,  _, momentum):
        self._out_shape = inp_shape
        self._mv_mean, self._mv_var = \
            server.issue_movingvar_slot(gamma_shape, momentum),\
            server.issue_movingvar_slot(gamma_shape, momentum)
        rank = len(inp_shape)
        f_dims = np.arange(rank)
        self._fd = tuple(f_dims)

    def forward(self, x, gamma, is_training):
        if is_training:
            mean = x.mean(self._fd)
            var = x.var(self._fd)
            self._mv_mean.apply_update(mean)
            self._mv_var.apply_update(var)
        else:
            mean = self._mv_mean.val
            var  = self._mv_var.val

        self._rstd = 1. / np.sqrt(var + 1e-8)
        self._normed = (x - mean) * self._rstd
        self._gamma = gamma
        return self._normed * gamma
    
    def backward(self, grad):
        N = np.prod(grad.shape[:-1])
        tmp = np.multiply(grad, self._normed).sum(self._fd)
        x_ = grad - self._normed * tmp * 1. / N
        x_ = self._rstd * self._gamma * x_
        return x_ - x_.mean(self._fd), tmp, None	

class add_biases(Module):
    def __init__(self, server, inp_shape, b_shape):
        rank = len(inp_shape)
        sum_over = np.arange(rank)
        self._range = tuple(sum_over)
        self._out_shape = inp_shape

    def forward(self, x, b):
        return x + b

    def backward(self, grad):
        return grad, grad.sum(self._range)
        
class drop(Module):
    def __init__(self, server, inp_shape, _):
        self._out_shape = inp_shape

    def forward(self, x, keep_prob):
        f = x.shape[-1]; self._keep = keep_prob
        self._r = binomial(1, self._keep, [1, f])
        self._r = self._r.astype(np.float64)
        return x * self._r / self._keep

    def backward(self, grad):
        grad_mask = grad / self._keep
        return grad_mask * self._r, None
    
    def unitest(self, *args):
        return self.forward(*args)