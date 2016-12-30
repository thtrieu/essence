import numpy as np
from module import Module
from src.utils import xavier, guass, nxshape
from recurring import Recurring, gate
from activations import *
        
class lstm_step(Recurring):
    # TODO: MAKE THIS A VALID USER MODULE
    def _setup(self, server, inp_shape, 
               hidden_size, forget_bias):
        time, emb = inp_shape
        w_shp = (hidden_size + emb, hidden_size)
        b_shp = (hidden_size,)
        self._gates = dict({
            'f': gate(server, w_shp, b_shp, sigmoid, forget_bias),
            'i': gate(server, w_shp, b_shp, sigmoid),
            'o': gate(server, w_shp, b_shp, sigmoid),
            'g': gate(server, w_shp, b_shp, tanh) })
        self._out_shape = b_shp
    
    def forward(self, c, hx):
        # hx is concatenation
        # of hidden and input
        gate_vals = list()
        for key in 'oifg': 
            gate_vals.append(
                self._gates[key].forward(hx))
        o, i, f, g = gate_vals
        c_new  = c * f + i * g
        squeeze = np.tanh(c_new)
        self._push((gate_vals, c, squeeze))
        h_new = o * squeeze
        return c_new, h_new
    
    def backward(self, gc, gh):
        ghx = 0. # return this
        gates = self._gates
        gate_vals, c, sqz = self._pop()
        o, i, f, g = gate_vals
        dsqz = 1 - sqz * sqz
        # linear carousel:
        gc_old = gh * o * dsqz + gc
        go, gi = gh * sqz, gc_old * g
        gf, gg = gc_old * c, gc_old * i
        for key, grad in zip('oifg', [go, gi, gf, gg]):
            ghx = ghx + self._gates[key].backward(grad)
        return gc_old * f, ghx
        

class lstm(Module):
    """ BASIC Long Short Term Memory """

    def __init__(self, server, inp_shape, lens_shape, 
                hidden_size, forget_bias):
        self._step = lstm_step(
            server, inp_shape, hidden_size, forget_bias)
        self._out_shape = (inp_shape[0], hidden_size)
        self._size = hidden_size

    def forward(self, x, lens = None):
        # create mask for lens
        full_len = x.shape[1]
        if lens is not None:
            lens = np.array(lens) - 1.0
        elif lens is None:
            lens = np.ones(full_len)
            lens = lens * full_len
        lens = lens[:, None]
        
        mask = (lens >= range(full_len))
        mask = mask.astype(np.float32)
        out_shape = nxshape(
            x, self._step._out_shape)

        o = np.zeros(out_shape)
        h = np.zeros(out_shape)
        c = np.zeros(out_shape)
        result = list() # return this
        for t in range(full_len):
            x_t = x[:, t, :]
            hx = np.concatenate([h, x_t], 1)
            c, h_new = self._step.forward(c, hx)
            mt = mask[:, t, None]
            h = (1 - mt) * o + mt * h_new
            result.append(h)
        self._mask = mask
        return np.stack(result, 1)
    
    def backward(self, gh): # b x t x s
        grad_x = list()
        grad_h = 0.; grad_c = 0.
        gh *= self._mask[:, :, None]

        for t in range(self._step.size(), 0, -1):
            grad_h += gh[:, t - 1, :]
            grad_c, grad_hx = \
                self._step.backward(grad_c, grad_h)
            grad_h = grad_hx[:, :self._size]
            gx = grad_hx[:, self._size:]
            grad_x = [gx] + grad_x

        return np.stack(grad_x, 1)