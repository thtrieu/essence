import numpy as np

class Attention(object):
    def __init__(self): pass
    def forward(self): pass
    def backward(self, grad): pass

class ContentBased(Attention):
    def forward(self, key, content): pass
    def backward(self, grad): pass

class LocationBased(Attention):
    def forward(self): pass
    def backward(self): pass

class cosine_sim(ContentBased):
    def forward(self, key, memory):
        dot = np.einsum('bnm,bm->bn', memory, key)
        norm_key = np.linalg.norm(key, axis = -1)
        norm_mem = np.linalg.norm(memory, axis = -1)
        norm = norm_key * norm_mem
        self._cache = (memory, key, norm)
        return dot / norm
    
    def backward(self, grad):
        memory, key, norm = self._cache
        grad_dot = grad / norm
        grad_mem = np.einsum('bn,bm->bnm', grad_dot, key)
        grad_key = np.einsum('bnm,bn->bm', memory, grad_dot)
        return grad_mem, grad_key

class interpolate(LocationBased):
    def forward(self, alpha, new, prev):
        self._cache = (alpha, new, prev)
        return alpha * new + (1-alpha) * prev
    
    def backward(self, grad):
        alpha, new, prev = self._cache
        grad_prev = grad * (1 - alpha)
        grad_new = grad * alpha
        grad_alpha = (grad * (new - prev)).sum(-1)
        return grad_new, grad_prev, grad_alpha
    
class rotate_conv(LocationBased):
    def forward(self, x, kernel):
        