from recurring import Recurring, gate
from activations import softplus, tanh, sigmoid, softmax
from mechanics import * 

class ntm_attend(Recurring):
    def _setup(self, server, lstm_size, vec_size, shift):
        self._gates = dict({
            'k': gate(server, (lstm_size, vec_size), None, sigmoid), # key gate
            'b': gate(server, (lstm_size, 1), 1., softplus), # strength gate
            'i': gate(server, (lstm_size, 1), None, sigmoid), # interpolate gate
            'r': gate(server, (lstm_size, 
                              2 * shift + 1), None, softmax), # rotate gate
            's': gate(server, (lstm_size, 1), 1., softplus) # sharpen gate
        })
        self._mechanic = dict({ # mechanics
            'cos': cosine_sim(), # similarity
            'soft': normalise(), # softmax 
            'inter': interpolate(), # interpolate
            'rotate': circular_conv(), # circular conv
            'sharp': sharpen(), # sharpen
        })
    
    def forward(self, memory, lstm_h, w_prev):
        gates_vals = list()
        for name in 'kbirs':
            gates_vals.append(
                self._gates[name].forward(lstm_h))
        k, b, i, r, s = gates_vals
        sim = self._mechanic['cos'].forward(memory, k)
        self._push(b, sim) # b x 1 and b x n
        w_c = self._mechanic['soft'].forward(b * sim)
        w_i = self._mechanic['inter'].forward(w_c, w_prev, i)
        w_r = self._mechanic['rotate'].forward(w_i, r)
        w_new = self._mechanic['sharp'].forward(w_r, s + 1.)
        # print sim.std(-1).mean(), w_c.std(-1).mean()
        # print w_r.std(-1).mean(), w_new.std(-1).mean()
        #print w_prev.std(-1).mean()
        return w_new
    
    def backward(self, g_wnew):
        g_wr, gs = self._mechanic['sharp'].backward(g_wnew)
        g_wi, gr = self._mechanic['rotate'].backward(g_wr)
        g_wc, g_wprev, gi = self._mechanic['inter'].backward(g_wi)

        g_bsim = self._mechanic['soft'].backward(g_wc)
        b, sim = self._pop()
        gb = (g_bsim * sim).sum(-1, keepdims = True)
        g_sim = g_bsim * b
        
        g_mem, gk = self._mechanic['cos'].backward(g_sim)
        
        g_h = 0.
        for gate, grad in zip('kbirs', [gk, gb, gi, gr, gs]):
            g_h = g_h + self._gates[gate].backward(grad)
        return g_mem, g_h, g_wprev