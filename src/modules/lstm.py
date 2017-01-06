import numpy as np
from .module import Module
from src.utils import xavier, guass, nxshape, rand
from .activations import *
from .rnn_step import lstm_step

class lstm(Module):
    """ BASIC Long Short Term Memory """

    def __init__(self, server, inp_shape, lens_shape, 
                hidden_size, forget_bias, 
                gate_activation, read_activation, 
                transfer):
        self._step = lstm_step(
            server, inp_shape, hidden_size, forget_bias, 
            gate_activation, read_activation, transfer)
        self._out_shape = (inp_shape[0], hidden_size)

    def forward(self, x, lens = None):
        # create mask for lens
        self._step.flush()
        full_len = x.shape[1]
        if lens is not None:
            lens = np.array(lens) - 1.0
        elif lens is None:
            lens = np.ones(full_len)
            lens = lens * full_len
        lens = lens[:, None]
        
        mask = (lens >= range(full_len))
        mask = mask.astype(np.float64)
        self._mask = mask[:, :, None]
        out_shape = nxshape(
            x, self._step._out_shape)
        # Loop through time steps
        h = np.zeros(out_shape)
        c = np.zeros(out_shape)
        result = list() # return this
        for t in range(full_len):
            c, h = self._step.forward(
                c, h, x[:, t, :])
            result.append(h)
        return np.stack(result, 1) * self._mask
    
    def backward(self, gh): # b x t x s
        grad_x = list()
        grad_h = 0.; grad_c = 0.
        gh *= self._mask

        for t in range(gh.shape[1] -1, -1, -1):
            grad_h += gh[:, t - 1, :]
            grad_c, grad_h, gx = \
                self._step.backward(grad_c, grad_h)
            grad_x = [gx] + grad_x
        
        return np.stack(grad_x, 1)