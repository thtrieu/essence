from recurring import Recurring, gate
from module import Module
from lstm import lstm_step
from activations import *
from attentions import ntm_attend 

class turing(Module):
    def __init__(self, server, x_shape, out_size, 
                memory_size, vec_size, lstm_size, 
                max_shift = 1):
       
        memory = ntm_memory()
        self._step = ntm_step()

    def forward(self):
        pass
