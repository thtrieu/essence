from module import Module
from src.utils import binomial
from conv import c_xpool2, c_gradxp2
import numpy as np

class reshape(Module):
    def _setup(self, new_shape):
        self._out_shape = (new_shape)

    def forward(self, x):
        return x.reshape(self.out_shape)
    
    def backward(self, grad):
        return grad.reshape(self.inp_shape)		

class add_biases(Module):
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
    def _setup(self, keep_prob = .5):
        self.keep = keep_prob

    def forward(self, x):
        f = x.shape[-1]
        self.r = binomial(1, self.keep, [1, f]).astype(np.float32)
        return x * self.r / self.keep

    def backward(self, grad):
        grad_mask = grad / self.keep
        return grad_mask * self.r

class pool(Module):
    pass

class maxpool2x2(pool):
    def _setup(self):
        h, w, f = self._inp_shape
        assert not h%2 and not w%2, \
        'pool2x2 on odd size not supported'
        self._out_shape = [h/2, w/2, f]

    def forward(self, x):
        pooled = np.zeros(self.out_shape, dtype = np.float32)
        self._mark = np.zeros(self.out_shape, dtype = np.int32)
        c_xpool2(x, self._mark, pooled, *self._inp_shape)
        return pooled

    def backward(self, grad):
        unpooled = np.zeros(self.inp_shape, dtype = np.float32)
        c_gradxp2(unpooled, self._mark, grad, *self._inp_shape)
        return unpooled

class pad(Module):    
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