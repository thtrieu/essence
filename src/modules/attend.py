from recurring import Recurring, gate
from activations import softmax 

class ntm_attend(Recurring):
    def _setup(self, server, lstm_size, vec_size, shift):
        self._gates = dict({
            'k': gate(server, (lstm_size, vec_size), None, tanh), # key
            'b': gate(server, (lstm_size, 1), None, softplus), # beta (strength)
            'i': gate(server, (lstm_size, 1), None, sigmoid), # interpolate
            'r': gate(server, (lstm_size, 2 * shift + 1), 
                      bias = None, act_class = softmax), # rotate conv
            's': gate(server, (lstm_size, 1), None, softplus) # sharpen
        })
    
    def forward(self, memory, lstm_h, w_prev):
        gates_vals = list()
        for name in 'kbirs':
            gates_vals.append(
                self._gates[name](lstm_h))
        k, b, i, r, s = gates_vals
        cos = cosine_sim(); soft = softmax();
        pol = interpolate(); rot = rotate_conv(); 
        norm = norm_softplus();
        self._push([cos, soft, pol, rot, norm])

        sim = cos.forward(k, memory)
        w_c = soft.forward(b * sim) # content-based attention
        w_i = pol.forward(i, w_c, w_prev) # i * w_c + (1-i) * w_prev
        w_r = rot.forward(w_i, r) # rotated attention
        w_new = norm.forward(w_r, 1 + s) # sharpen attention
        return w_new
    
    def backward(self, g_wnew):
        cos, soft, rot, norm = self._pop()
        g_wr, gs = norm.backward(g_wnew)
        g_wi, gr = rot.backward(g_wr)
        g_wc, g_wprev, gi = pol.backward(g_wi)
        g_sim, gb = soft.backward(g_wc)
        g_mem, gk = cos.backward(g_sim)
        
        g_h = 0.
        for gate, grad in zip('kbgrs', [gk, gb, gi, gr, gs]):
            g_h = gh + self._gates[gate].backward(grad)
        return g_wprev, g_mem, g_h