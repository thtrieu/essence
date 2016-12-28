from conv import c_conv2d, c_gradk, c_gradx
from src.utils import nxshape
from module import Module
import numpy as np

class matmul(Module):    
    def __init__(self, server, x_shape, w_shape):
        self._out_shape = (w_shape[1],)

    def forward(self, x, w):
        self._x = x
        self._w = w
        return x.dot(w)

    def backward(self, grad):
        dx = grad.dot(self._w.T)
        dw = self._x.T.dot(grad)
        return dx, dw

class conv(Module):
    def __init__(self, serve, x_shape, k_shape, pad, stride):
        kh, kw, f, f_ = k_shape
        self._kshape = k_shape
        self._xshape = x_shape
        h, w, _ = x_shape
        sh, sw = stride
        ph, pw = pad

        self._args = [
            h, w, f, ph, pw, 
            f_, kh, kw, sh, sw]

        h_ = (h + 2*ph - kh) / sh + 1
        w_ = (h + 2*pw - kw) / sw + 1
        self._out_shape = [h_, w_, f_]

    def forward(self, x, k):
        self._x = x; self._k = k
        shp = nxshape(x, self._out_shape)
        conved = conv.zeros32(shp)
        c_conv2d(x, k, conved, *self._args)
        return conved
    
    def backward(self, grad):
        grad = grad.astype(np.float32)
        gk = conv.zeros32(self._kshape)
        c_gradk(self._x, gk, grad, *self._args)
        
        xshp = nxshape(grad, self._xshape)
        gx = conv.zeros32(xshp)
        c_gradx(gx, self._k, grad, *self._args)
        return gx, gk
    
    @classmethod
    def zeros32(cls, shape):
        return np.zeros(shape, dtype = np.float32)