from rnn_mem import ntm_memory
from attentions import ntm_attend
from recurring import Recurring, gate
from activations import *
import numpy as np

# TODO: make all the steps 
# available to end users
        
class lstm_step(Recurring):
    def _setup(self, server, inp_shape, 
               hidden_size, forget_bias):
        _, emb = inp_shape
        w_shp = (hidden_size + emb, hidden_size)
        self._gates = dict({
            # gate args: server, shape, bias, act
            'f': gate(server, w_shp, forget_bias),
            'i': gate(server, w_shp, None, sigmoid),
            'o': gate(server, w_shp, None, sigmoid),
            'g': gate(server, w_shp, None, tanh) })
        self._out_shape = (hidden_size,)
        self._size = hidden_size
    
    def forward(self, c, h, x):
        hx = np.concatenate([h, x], 1)
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
        gates = self._gates
        gate_vals, c, sqz = self._pop()
        o, i, f, g = gate_vals
        dsqz = 1 - sqz * sqz
        
        # linear carousel:
        gc_old = gh * o * dsqz + gc
        go, gi = gh * sqz, gc_old * g
        gf, gg = gc_old * c, gc_old * i

        ghx = 0.
        for key, grad in zip('oifg', [go, gi, gf, gg]):
            ghx = ghx + self._gates[key].backward(grad)
        gh, gx = ghx[:, :self._size], ghx[:, self._size:]
        return gc_old * f, gh, gx

class ntm_step(Recurring):
    def __init__(self, server, x_shape, mem_size, vec_size, 
                 lstm_size, shift, out_size):
        time_length, x_size = x_shape
        control_xshape = (time_length, x_size + lstm_size + vec_size)
        self._control = lstm_step(server, control_xshape, lstm_size, 1.5)
        self._rhead = ntm_attend(server, lstm_size, vec_size, shift)
        self._whead = ntm_attend(server, lstm_size, vec_size, shift)
        self._memory = ntm_memory(lstm_size, mem_size, vec_size)
        self._readout = gate(server, (lstm_size, out_size))

    
    def forward(self, c, h_nmt, x, w_read, w_write, memory):
        c, h_lstm = self._control.forward(c, h_nmt, x)
        w_read = self._rhead.forward(memory, h_lstm, w_read)
        w_write = self._whead.forward(memory, h_lstm, w_write)
        mem_read, new_mem = self._memory.forward(
            h_lstm, w_read, w_write, memory)
        h_nmt = np.concatenate([mem_read, h_lstm], 1)
        readout = self._readout.forward(h_lstm)
        recurlets = (c, h_nmt, w_read, w_write, new_mem)
        return recurlets, readout
    
    def backward(self, gc, gh_nmt, gr, gw, gm, gout):
        """
        Args: 
            Respectively, grad of lstm's cell, h_nmt, 
            w_read, w_write, new_memory, & readout
        """
        g_memread = gh_nmt[:, :self._vec_size]
        g_hlstm = gh_nmt[:, self._vec_size:]
        g_hout = self._readout.backward(gout)
        # grad flow through memory
        gm, gm_r, gm_w, gm_h = \
            self._memory.backward(g_memread, gm)
        # grad flow through write & read attention
        gw, gw_m, gw_h = self._whead.backward(gw)
        gr, gr_m, gr_h = self._rhead.backward(gr)
        # grad flow through controller
        gc, gh, gx = self._control.backward(
            gc, g_hlstm + ghout + gm_h + gw_h + gr_h)
        # grad summing
        gm = gm + gr_m + gw_m
        gw, gr = gw + gm_w, gr + gm_r
        return gc, gh, gr, gw, gm, gx