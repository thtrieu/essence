from .rnn_mem import ntm_memory
from .attention import ntm_attend
from .recurring import Recurring, gate
from .activations import *
import numpy as np

# TODO: make all the steps available to end users
        
class lstm_step(Recurring):
    def _setup(self, server, inp_shape, 
               hidden_size, forget_bias, 
               gate_activation, 
               read_activation, 
               transfer):

        _, emb = inp_shape
        w_shp = (hidden_size + emb, hidden_size)
        if transfer is None: transfer = [None] * 4
        gact = activation_dict[gate_activation]
        ract = activation_dict[read_activation]
        self._gates = dict({
            # gate args: server, shape, bias, act
            'f': gate(server, w_shp, forget_bias, gact, transfer[0]),
            'i': gate(server, w_shp, None, gact, transfer[1]),
            'o': gate(server, w_shp, None, gact, transfer[2]),
            'g': gate(server, w_shp, None, ract, transfer[3]) })
        self._out_shape = (hidden_size,)
        self._size = hidden_size
    
    def _flush(self):
        for gate_name in self._gates:
            self._gates[gate_name].flush()
    
    def forward(self, c, h, x):
        hx = np.concatenate([h, x], 1)
        gate_vals = list()
        for key in 'oifg': 
            gate_vals.append(
                self._gates[key].forward(hx))
        o, i, f, g = gate_vals
        c_new  = c * f + i * g
        squeeze = np.tanh(c_new)
        self._push(gate_vals, c, squeeze)
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
    def _setup(self, server, x_shape, mem_size, vec_size, 
                 lstm_size, shift, out_size):
        self._vec_size = vec_size # M
        x_size = x_shape[-1]
        control_xshape = (None, vec_size + x_size)
        self._control = lstm_step(server, control_xshape, lstm_size, 1.5)
        self._rhead = ntm_attend(server, lstm_size, vec_size, shift)
        self._whead = ntm_attend(server, lstm_size, vec_size, shift)
        self._memory = ntm_memory(server, lstm_size, mem_size, vec_size)
        self._readout = gate(server, (lstm_size, out_size), None, sigmoid)
    
    def _flush(self):
        self._control.flush()
        self._rhead.flush()
        self._whead.flush()
        self._memory.flush()
        self._readout.flush()

    def forward(self, c, h, x, w_read, w_write, mem_read, memory):
        mx = np.concatenate([mem_read, x], 1)
        c_new, h_new = self._control.forward(c, h, mx)
        new_w_read = self._rhead.forward(memory, h_new, w_read)
        new_w_write = self._whead.forward(memory, h_new, w_write)
        new_mem_read, new_memory = self._memory.forward(
            h_new, new_w_read, new_w_write, memory)
        readout = self._readout.forward(h_new)
        return c_new, h_new, new_w_read, new_w_write, \
               new_mem_read, new_memory, readout
    
    def backward(self, gc, gh, gr, gw, gread, gm, gout):
        """
        Args: 
            Respectively, gradient of lstm's cell, h, 
            w_read, w_write, new_memory, mem_read & readout
        """
        # grad flow through readout
        g_hout = self._readout.backward(gout)
        # grad flow through memory
        gm_h, gm_r, gm_w, gm = \
            self._memory.backward(gread, gm)
        # grad flow through write & read attention
        gw_m, gw_h, gw = self._whead.backward(gw + gm_w)
        gr_m, gr_h, gr = self._rhead.backward(gr + gm_r)
        # grad flow through controller
        gc, gh, gx = self._control.backward(
            gc, gh + g_hout + gm_h + gw_h + gr_h)
        gread = gx[:, :self._vec_size]
        gx = gx[:, self._vec_size:]
        # grad summation
        gm = gm + gr_m + gw_m
        return gc, gh, gx, gr, gw, gread, gm