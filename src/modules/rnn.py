import numpy as np
from module import Module
from src.utils import xavier, guass, nxshape

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _dsigmoid(x):
    return x * (1. - x)

def _dtanh(x):
    return 1. - x * x

class lstm_uni(Module):
    """ Many-to-one BASIC Long Short Term Memory """

    class step(object):
        activate = dict({
            'f': _sigmoid, 'i': _sigmoid,
            'o': _sigmoid, 'g': np.tanh })

        dactivate = dict({
            'f': _dsigmoid, 'i': _dsigmoid,
            'o': _dsigmoid, 'g': _dtanh })

        def __init__(self, p):
            self._p = p
            self._gates = dict()
        
        def _gate_forward(self, key):
            w, b = self._p[key]
            tmp = self.hx.dot(w.val) + b.val
            act = self.activate[key](tmp)
            self._gates[key] = act
        
        def _gate_backward(self, grad, key):
            gate_key = self._gates[key]
            partial = self.dactivate[key](gate_key)
            grad = grad * partial

            w, b = self._p[key]
            w.set_grad(self.hx.T.dot(grad))
            b.set_grad(grad.sum())
            return grad.dot(w.val.T)

        def forward(self, c, hx):
            self.c = c; self.hx = hx
            for key in 'oifg': self._gate_forward(key)
            gates = self._gates
            c_new = gates['f'] * c 
            c_new += gates['i'] * gates['g']
            self._tanh = np.tanh(c_new)
            h_new = gates['o'] * self._tanh
            return c_new, h_new
        
        def backward(self, gc, gh):
            gates = self._gates
            tanh2 = self._tanh * self._tanh
            g_tanh = gh * gates['o']
            ghx = 0.
            # linear carousel:
            gc_ = g_tanh * (1. - tanh2) + gc
            go, gi = gh * self._tanh, gc_ * gates['g']
            gf, g_ = gc_ * self.c, gc_ * gates['i']
            for pair in zip([go, gi, gf, g_], 'oifg'):
                ghx += self._gate_backward(*pair)
            return gc_ * gates['f'], ghx

    def _gate_var(self, server, w_shape, 
                    b_shape, bias = None):
        w_init = xavier(w_shape)
        if bias is not None:
            b_init = np.ones(b_shape) * bias
        elif bias is None:
            b_init = guass(0., 1e-2, b_shape)
        w_slot = server.issue_var_slot(w_init, True) 
        b_slot = server.issue_var_slot(b_init, True) 
        return w_slot, b_slot

    def __init__(self, server, inp_shape, lens_shape, 
                hidden_size, forget_bias):
        self._max_len, emb_dim = inp_shape
        w_shp = (hidden_size + emb_dim, hidden_size)
        b_shp = (hidden_size,)
        self._p = dict({
            'f': self._gate_var(server, w_shp, b_shp, forget_bias),
            'i': self._gate_var(server, w_shp, b_shp),
            'o': self._gate_var(server, w_shp, b_shp),
            'g': self._gate_var(server, w_shp, b_shp) })
        self._out_shape = (hidden_size,)
        self._size = hidden_size

    def forward(self, x, lens):
        lens = np.array(lens) 
        self._pad = self._max_len - lens.max()
        onehot = np.zeros((lens.size, lens.max()))
        onehot[np.arange(lens.size), lens - 1] = 1
        out_shape = nxshape(x, self._out_shape)

        result = np.zeros(out_shape)
        h = np.zeros(out_shape)
        c = np.zeros(out_shape)

        self._unrolled = list()
        for t in range(lens.max()):
            x_t = x[:, t, :]
            new_step = lstm_uni.step(self._p)
            hx = np.concatenate([h, x_t], 1)
            c, h = new_step.forward(c, hx)
            result += onehot[:, t, None] * h
            self._unrolled.append(new_step)
        self._1hot = onehot
        return result

    def backward(self, gh, gc = 0.):
        grad_x = list()
        grad_h = 0.; grad_c = 0.
        for t in range(len(self._unrolled), 0, -1):
            lstm_step = self._unrolled[t - 1]
            grad_h += self._1hot[:, t - 1, None] * gh
            grad_c, grad_hx = \
                lstm_step.backward(grad_c, grad_h)
            grad_h = grad_hx[:,:self._size]
            gx = grad_hx[:,self._size:]
            grad_x = [gx] + grad_x
        grad_x = np.stack(grad_x, 1)
        return np.pad(grad_x, 
            ((0,0),(0,self._pad),(0,0)), 'constant')