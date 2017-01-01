from module import Module
from rnn_step import ntm_step
from src.utils import nxshape
import numpy as np

class turing(Module):
    def __init__(self, server, x_shape, out_size, 
                memory_size, vec_size, 
                lstm_size, shift = 1):
        self._mem_shape = (memory_size, vec_size)
        self._read_size = (vec_size,)
        self._h_shape = (lstm_size,)
        self._w_shape = (memory_size,)
        self._out_shape = (x_shape[0], out_size)
        self._step = ntm_step(
            server, x_shape, memory_size, vec_size, 
            lstm_size, shift, out_size)

    def forward(self, x, lens = None):
        # Initial state
        memory = np.zeros(nxshape(x, self._mem_shape))
        mem_read = np.random.rand(*nxshape(x, self._read_size)) * .1
        h = np.zeros(nxshape(x, self._h_shape))
        c = np.zeros(h.shape)
        w_read = np.zeros(nxshape(x, self._w_shape))
        w_write = np.zeros(nxshape(x, self._w_shape))
        w_read[:, 0] = np.ones(w_read.shape[0])
        w_write[:, -1] = np.ones(w_write.shape[0])
        
        # Loop
        result = list()
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            recurlets, readout = self._step.forward(
                c, h, x[:, t, :], w_read, 
                w_write, mem_read, memory)
            c, h_new, w_read, w_write, \
            mem_read, memory = recurlets
            result.append(readout)
        return np.stack(result, 1)

    def backward(self, grad):
        gread = np.zeros(nxshape(grad, self._read_size))
        gmem = np.zeros(nxshape(grad, self._mem_shape))
        gh = np.zeros(nxshape(grad, self._h_shape))
        gr = np.zeros(nxshape(grad, self._w_shape))
        gc = np.zeros(gh.shape)
        gw = np.zeros(gr.shape)

        gradx = list();
        for t in range(grad.shape[1] - 1, - 1, - 1):
            grad_t = grad[:, t, :]
            gc, gh, gr, gw, gread, gmem, gx_t = \
                self._step.backward(gc, gh, gr, gw,
                    gread, gmem, grad_t)
            gradx = [gx_t] + gradx
        return np.stack(gradx, 1)