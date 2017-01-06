from .conv import c_conv2d, c_gradk, c_gradx, c_xpool2, c_gradxp2
from src.utils import nxshape
from .module import Module
import numpy as np

class matmul(Module):
    def __init__(self, server, x_shape, w_shape):
        self._inp_shape = x_shape
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
    def __init__(self, server, x_shape, k_shape, pad, stride):
        kh, kw, f, f_ = k_shape
        self._kshape = k_shape
        self._xshape = x_shape
        h, w, _ = x_shape
        sh, sw = stride
        ph, pw = pad

        self._args = [
            h, w, f, ph, pw, 
            f_, kh, kw, sh, sw]

        h_ = (h + 2*ph - kh) // sh + 1
        w_ = (h + 2*pw - kw) // sw + 1
        self._out_shape = [h_, w_, f_]

    def forward(self, x, k):
        self._x = x; self._k = k
        shp = nxshape(x, self._out_shape)
        conved = np.zeros(shp)
        c_conv2d(x, k, conved, *self._args)
        return conved
    
    def backward(self, grad):
        gk = np.zeros(self._kshape)
        c_gradk(self._x, gk, grad, *self._args)
        xshp = nxshape(grad, self._xshape)
        gx = np.zeros(xshp)
        c_gradx(gx, self._k, grad, *self._args)
        return gx, gk


class maxpool2x2(Module):
    def __init__(self, server, inp_shape):
        h, w, f = inp_shape
        self._inp_shape = inp_shape
        assert not h % 2 or not w % 2, \
        'pool2x2 on odd size not supported'
        self._out_shape = [h // 2, w // 2, f]

    def forward(self, x):
        pooled = np.zeros(nxshape(x, self._out_shape))
        self._mark = np.zeros(nxshape(x, self._out_shape))
        self._mark = self._mark.astype(np.int32)
        c_xpool2(x, self._mark, pooled, *self._inp_shape)
        return pooled

    def backward(self, grad):
        n, h, w, f = grad.shape
        unpooled = np.zeros((n, h*2, w*2, f))
        c_gradxp2(unpooled, self._mark, 
                  grad, *self._inp_shape)
        return unpooled
    
    def unitest(self, x):
        return self.forward(x)