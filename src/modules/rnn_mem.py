from recurring import Recurring, gate
import numpy as np
from activations import *

class ntm_memory(Recurring):
    def _setup(self, server, lstm_size, mem_size, vec_size):
        self._size = (mem_size, vec_size)
        self._gates = dict({
            'f': gate(server, (lstm_size, vec_size), None, sigmoid), # forget
            'a': gate(server, (lstm_size, vec_size), None, tanh) # add
        })
    
    def forward(self, h_lstm, w_read, w_write, memory):
        # read
        mem_read = np.einsum('bnm,bn->bm', memory, w_read)
        # erase & add
        e = self._gates['f'].forward(h_lstm) # vec_size
        a = self._gates['a'].forward(h_lstm) # vec_size
        erase = w_write[:, :, None] * e[:, None, :]
        write = w_write[:, :, None] * a[:, None, :]
        erase_comp = 1. - erase
        new_mem = memory * erase_comp + write
        # push to recurrent stack
        self._push(
            memory, e, a, 
            w_read, w_write, 
            erase_comp)
        return mem_read, new_mem
    
    def backward(self, g_memread, g_newmem):
        memory, e, a, w_read, w_write, erase_comp = self._pop()
        gm = np.einsum('bn,bm->bnm', w_read, g_memread)
        gr = np.einsum('bnm,bm->bn', memory, g_memread)

        # grad flow through new_mem
        ge_m = g_newmem * erase_comp
        g_erase = -1. * g_newmem * memory
        # grad flow through eraser
        g_e = np.einsum('bnm,bn->bm', g_erase, w_write)
        g_ew = np.einsum('bnm,bm->bn', g_erase, e)
        # grad flow through writer
        g_a = np.einsum('bnm,bn->bm', g_newmem, w_write)
        g_aw = np.einsum('bnm,bm->bn', g_newmem, a)
        # grad flow though gates
        g_he = self._gates['f'].backward(g_e)
        g_ha = self._gates['a'].backward(g_a)
        # summing grads
        gm = gm + ge_m
        gw = g_ew + g_aw
        gh = g_he + g_ha
        return gm, gr, gw, gh