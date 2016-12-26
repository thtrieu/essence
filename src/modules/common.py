from module import Module, ChainModule
from src.utils import binomial
from conv import c_xpool2, c_gradxp2
import numpy as np

class batch_norm(Module):
    def _prepare(self, inp_shape, _, momentum = .9):
        self._out_shape = inp_shape[1:]
        self._gamma = self._var('gamma')
        self._mv_mean = self._var('mean')
        self._mv_var = self._var('var')
        self._alpha = momentum
        rank = len(inp_shape)-1
        f_dims = np.arange(rank)
        self._fd = tuple(f_dims)
    
    def _update_mv_ave(self, v, v_):
        _alpha = 1. - self._alpha
        return v * self._alpha + _alpha * v_

    def forward(self, x, is_training):
        if is_training:
            mean = x.mean(self._fd)
            var = x.var(self._fd)
            self._mv_mean.apply_update(
                self._update_mv_ave, mean)
            self._mv_var.apply_update(
                self._update_mv_ave, var)
        else:
            mean = self._mv_mean.val
            var  = self._mv_var.val

        self._rstd = 1. / np.sqrt(var + 1e-8)
        self._normed = (x - mean) * self._rstd
        return self._normed * self._gamma.val
    
    def backward(self, grad):
        N = np.prod(grad.shape[:-1])

        tmp = np.multiply(grad, self._normed).sum(self._fd)
        self._gamma.set_grad(tmp.sum) # gradient for gamma
        x_ = grad - self._normed * tmp * 1. / N
        x_ = self._rstd * self._gamma.val * x_
        mean, var = x_.mean(self._fd), x_.var(self._fd)
        return (x_ - mean) / np.sqrt(var + 1e-8)

class reshape(ChainModule):
    def _setup(self, new_shape):
        self._out_shape = (new_shape)

    def forward(self, x):
        return x.reshape(
            self.out_shape(x.shape[0]))
    
    def backward(self, grad):
        return grad.reshape(
            self.inp_shape(grad.shape[0]))		

class add_biases(ChainModule):
    def _setup(self, *args, **kwargs):
        self._b = self._var()
        rank = len(self._inp_shape)
        sum_over = np.arange(rank)
        self._range = tuple(sum_over)

    def forward(self, x):
        return x + self._b.val

    def backward(self, grad):
        self._b.set_grad(grad.sum, self._range)
        return grad
        
class drop(Module):
    def _prepare(self, shape, _):
        self._out_shape = shape

    def forward(self, x, keep_prob):
        f = x.shape[-1]; self._keep = keep_prob
        self.r = binomial(1, self._keep, [1, f]).astype(np.float32)
        return x * self.r / self._keep

    def backward(self, grad):
        grad_mask = grad / self._keep
        return grad_mask * self.r

class maxpool2x2(ChainModule):
    def _setup(self):
        h, w, f = self._inp_shape
        assert not h%2 and not w%2, \
        'pool2x2 on odd size not supported'
        self._out_shape = [h/2, w/2, f]

    def forward(self, x):
        n = x.shape[0]
        pooled = np.zeros(self.out_shape(n), dtype = np.float32)
        self._mark = np.zeros(self.out_shape(n), dtype = np.int32)
        c_xpool2(x, self._mark, pooled, *self._inp_shape)
        return pooled

    def backward(self, grad):
        n = grad.shape[0]
        unpooled = np.zeros(self.inp_shape(n), dtype = np.float32)
        c_gradxp2(unpooled, self._mark, grad, *self._inp_shape)
        return unpooled

class pad(ChainModule):    
    def _setup(self, pad):
        self._pad = pad
        self._h0 = h0 = pad[0][0]
        self._w0 = w0 = pad[1][0]
        h1 = pad[0][1]
        w1 = pad[1][1]

        h, w, f = self._inp_shape
        self._out_shape = (
            h + h0 + h1, w + w0 + w1, f)

    def forward(self, x):
        pads = [(0,0)] + list(self._pad) + [(0,0)]
        return np.pad(x, pads, 'constant')

    def backward(self, grad):
        h, w, _ = self._inp_shape
        h0, w0 = self._h0, self._w0
        return grad[:, h0 : h0 + h, w0 : w0 + w, :]