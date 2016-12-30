import numpy as np
from module import Module
from src.utils import xavier, guass, nxshape
from recurring import Recurring, gate
from activations import *
        
class lstm_step(Recurring):
    def _setup(self, gates):
        self._gates = gates
    
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

class lstm_uni(Module):
    """ Many-to-one BASIC Long Short Term Memory """

    def __init__(self, server, inp_shape, lens_shape, 
                hidden_size, forget_bias):
        self._max_len, emb_dim = inp_shape
        w_shp = (hidden_size + emb_dim, hidden_size)
        b_shp = (hidden_size,)
        gates = dict({
            'f': gate(server, w_shp, b_shp, sigmoid, 1.5),
            'i': gate(server, w_shp, b_shp, sigmoid),
            'o': gate(server, w_shp, b_shp, sigmoid),
            'g': gate(server, w_shp, b_shp, tanh) })
        self._step = lstm_step(gates)
        self._out_shape = (hidden_size,)
        self._size = hidden_size

    def forward(self, x, lens):
        lens = np.array(lens) 
        self._pad = self._max_len - lens.max()
        onehot = np.zeros((lens.size, lens.max()))
        onehot[np.arange(lens.size), lens - 1] = 1
        out_shape = nxshape(x, self._out_shape)

        h = np.zeros(out_shape)
        c = np.zeros(out_shape)
        result = np.zeros(out_shape)
        for t in range(lens.max()):
            x_t = x[:, t, :]
            hx = np.concatenate([h, x_t], 1)
            c, h = self._step.forward(c, hx)
            result += onehot[:, t, None] * h
        self._1hot = onehot
        return result

    def backward(self, gh, gc = 0.):
        grad_x = list()
        grad_h = 0.; grad_c = 0.
        for t in range(self._step._size(), 0, -1):
            grad_h += self._1hot[:, t - 1, None] * gh
            grad_c, grad_hx = \
                self._step.backward(grad_c, grad_h)
            grad_h = grad_hx[:, :self._size]
            gx = grad_hx[:, self._size:]
            grad_x = [gx] + grad_x
        grad_x = np.stack(grad_x, 1)
        return np.pad(grad_x, 
            ((0,0), (0, self._pad), (0,0)), 'constant')