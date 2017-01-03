from recurring import Recurring
import numpy as np

class norm(Recurring):
    def forward(self, x):
        x2 = x * x
        x2_sum = x2.sum(-1, keepdims = True)
        x2_sqrt = np.sqrt(x2_sum) + 1e-8
        x_norm = x / x2_sqrt
        self._push(x2_sqrt, x_norm)
        return x_norm
    
    def backward(self, grad):
        x2_sqrt, x_norm = self._pop()
        g = (grad * x_norm).sum(-1, keepdims = True)
        g = grad - x_norm * g
        return g / x2_sqrt

class cosine_sim(Recurring):
    def _setup(self):
        self._knorm = norm()
        self._mnorm = norm()

    def forward(self, memory, key):
        knorm = self._knorm.forward(key)
        mnorm = self._mnorm.forward(memory)
        self._push(knorm, mnorm)
        result = np.einsum('bnm,bm->bn', mnorm, knorm)
        return result
    
    def backward(self, grad):
        knorm, mnorm = self._pop()
        gmem = np.einsum('bn,bm->bnm', grad, knorm)
        gkey = np.einsum('bnm,bn->bm', mnorm, grad)
        gk = self._knorm.backward(gkey)
        gm = self._mnorm.backward(gmem)
        return gm, gk

class normalise(Recurring):
    def forward(self, x):
        row_max = x.max(-1, keepdims = True)
        e_x = np.exp(x - row_max)
        e_sum = e_x.sum(-1, keepdims = True)
        result = e_x / (e_sum + 1e-8)
        self._push(result)
        return result
        
    def backward(self, grad):
        a = self._pop()
        m = np.multiply(grad, a)
        g = grad - m.sum(-1, keepdims = True)
        return np.multiply(g, a)

class interpolate(Recurring):
    def forward(self, new, prev, alpha):
        self._push(new, prev, alpha)
        result = alpha * new + (1. - alpha) * prev
        return result
    
    def backward(self, grad):
        new, prev, alpha = self._pop()
        grad_alpha = (grad * (new - prev)).sum(-1, keepdims = True)
        grad_prev = grad * (1. - alpha)
        grad_new = grad * alpha
        return grad_new, grad_prev, grad_alpha
    
class circular_conv(Recurring):
    def forward(self, x, kernel):
        s = kernel.shape[1] / 2
        k = np.arange(-s, s + 1)
        patches = list()
        M = x.shape[-1]
        for i in xrange(M):
            patches += [x[:, (k + i) % M]]
        self._push(patches, kernel, k)
        cols = list()
        for p in patches:
            col = (kernel * p).sum(-1, keepdims = True)
            cols.append(col)
        result = np.concatenate(cols, 1)
        return result

    def backward(self, grad):
        patches, kernel, k = self._pop()
        M = grad.shape[-1]
        gradx = np.zeros(grad.shape)
        gradk = np.zeros(kernel.shape)
        for i in xrange(M):
            gi = grad[:, i][:, None]
            gradx[:, (k + i) % M] += gi * kernel
            gradk += gi * patches[i]
        return gradx, gradk

class amplify(Recurring):
    def forward(self, x, gamma):
        result = np.power(x, gamma)
        self._push(x, result, gamma)
        return result

    def backward(self, grad):
        x, result, gamma = self._pop()
        gy = grad * result # y = gamma * ln x
        ggamma = (gy * np.log(x)).sum(
            -1, keepdims = True)
        return gamma * gy / (x + 1e-8), ggamma

class average(Recurring):
    def forward(self, x):
        x_sum = x.sum(-1, keepdims = True) + 1e-8
        result = x / x_sum
        self._push(x_sum, result)
        return result

    def backward(self, grad):
        x_sum, result = self._pop()
        g = grad * result
        g = grad - g.sum(-1, keepdims = True)
        return g / x_sum

class sharpen(Recurring):
    def _setup(self):
        self._amp = amplify()
        self._ave = average()

    def forward(self, x, gamma):
        xpow = self._amp.forward(x, gamma)
        xave = self._ave.forward(xpow)
        return xave 
    
    def backward(self, grad):
        gpow = self._ave.backward(grad)
        gx, ggamma = self._amp.backward(gpow)
        return gx, ggamma