from .module import Module
from .rnn_step import ntm_step
from src.utils import nxshape, guass, xavier, randint
import numpy as np

class turing(Module):
    def __init__(self, server, x_shape, out_size, 
                memory_size, vec_size, 
                lstm_size, shift = 1):
        self._mem_shape = (memory_size, vec_size)
        self._mem = np.ones(self._mem_shape) * .1
        self._read_size = (vec_size,)
        self._h_shape = (lstm_size,)
        self._w_shape = (memory_size,)
        self._out_shape = (x_shape[0], out_size)
        self._step = ntm_step(
            server, x_shape, memory_size, vec_size, 
            lstm_size, shift, out_size)

    def forward(self, x):
        self._step.flush()
        memory = np.array([self._mem] * x.shape[0])
        h = np.zeros(nxshape(x, self._h_shape))
        w_read = np.zeros(nxshape(x, self._w_shape))
        w_write = np.zeros(nxshape(x, self._w_shape))
        w_read[:, 0] = np.ones(w_read.shape[0])
        w_write[:, 0] = np.ones(w_write.shape[0])
        mem_read = memory[:,0,:]
        c = np.zeros(h.shape)
        
        # Loop through time
        result = list()
        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            c, h_new, w_read, w_write, \
            mem_read, memory, readout = \
            self._step.forward(
                c, h, x[:, t, :], w_read, 
                w_write, mem_read, memory)
            result.append(readout)
        
        result = np.stack(result, 1)
        return result

    def backward(self, grad):
        gread = np.zeros(nxshape(grad, self._read_size))
        gmem = np.zeros(nxshape(grad, self._mem_shape))
        gh = np.zeros(nxshape(grad, self._h_shape))
        gr = np.zeros(nxshape(grad, self._w_shape))
        gc = np.zeros(gh.shape)
        gw = np.zeros(gr.shape)

        gradx = list();
        for t in range(grad.shape[1] - 1, -1, -1):
            grad_t = grad[:, t, :]
            gc, gh, gx_t, gr, gw, gread, gmem = \
                self._step.backward(
                    gc, gh, gr, gw, gread, gmem, grad_t)
            gradx = [gx_t] + gradx
            
        return np.stack(gradx, 1)