from conv import c_conv2d, c_gradk, c_gradx
from module import Module, ChainModule
import numpy as np

class matmul(ChainModule):    
    def _setup(self, *args, **kwargs):
        self._w = self._var()
        self._out_shape = (self._w.val.shape[1],)

    def forward(self, x):
        self._x = x
        return x.dot(self._w.val)
    
    @classmethod
    def _cal_w_grad(cls, x, g):
        return x.transpose().dot(g)

    def backward(self, grad):
        self._var().set_grad( 
            matmul._cal_w_grad, self._x, grad)
        return grad.dot(self._w.val.transpose())

class conv(ChainModule):
    def _setup(self, pad, stride):
        ph, pw = pad; sh, sw = stride
        self._kernel = self._var()
        self._kshape = self._kernel.val.shape
        kh, kw, f, f_ = self._kshape
        h, w, _ = self._inp_shape
        self._args = [h, w, f, ph, pw, f_, kh, kw, sh, sw]

        h_ = (h + 2*ph - kh) / sh + 1
        w_ = (h + 2*pw - kw) / sw + 1
        self._out_shape = [h_, w_, f_]

    def forward(self, x):
        self._x = x; n = x.shape[0]
        conved = conv.zeros32(self.out_shape(n))
        c_conv2d(x, self._kernel.val, 
            conved, *self._args)
        return conved
    
    def _cal_kernel_grad(self, x, g):
        grad_kernel = conv.zeros32(self._kshape)
        c_gradk(x, grad_kernel, g, *self._args)
        return grad_kernel

    def backward(self, grad):
        grad = grad.astype(np.float32)
        self._kernel.set_grad(
            self._cal_kernel_grad, self._x, grad)
        
        n = grad.shape[0]
        grad_volume = conv.zeros32(self.inp_shape(n))
        c_gradx(
            grad_volume, self._kernel.val, grad, *self._args)
        return grad_volume
    
    @classmethod
    def zeros32(cls, shape):
        return np.zeros(shape, dtype = np.float32)