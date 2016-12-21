import numpy as np

class Module(object):
	def __init__(self, slot, *args, **kwargs):
		self._slot = slot
		self._setup(*args, **kwargs)

	def forward(self, x): pass 
	def backward(self, grad): pass
	def _cal_grad(self, *args): pass
	def _setup(self, *args, **kwargs): pass

class reshape(Module):
	def _setup(self, shape):
		self.shape = shape
		self.old = None

	def forward(self, x):
		self.old = x.shape
		return x.reshape(self.shape)
	
	def backward(self, grad):
		return grad.reshape(self.old)		

class add_biases(Module):
    def forward(self, x):
    	b = self._slot.val('b')
        return x + b

    def backward(self, grad):
        self._slot.set_grad('b', grad.sum, 0)
        return grad

class matmul(Module):
	def forward(self, x):
		self._x = x
		w = self._slot.val('w')
		return x.dot(w)

	def _cal_grad(self, x, g):
		return x.transpose().dot(g)

	def backward(self, grad):
		self._slot.set_grad('w', 
			self._cal_grad, self._x, grad)
		return grad.dot(
			self._slot.val('w').transpose())

class drop(Module):
	def _setup(self, keep_prob = .5):
		self.keep = keep_prob

	def forward(self, x):
		f = x.shape[-1]
		self.r = np.random.binomial(
			1, self.keep, [1, f])
		return x * self.r / self.keep

	def backward(self, grad):
		grad_mask = grad / self.keep
		return grad_mask * self.r

class pad(Module):
	def forward(self, x):
		pass
	def backward(self, grad):
		pass

class conv(module):
	def _setup(self, ksize, stride, kernel):
	"""
	Args:
		pad = ((ph1, ph2), (pw1, pw2))
		ksize = (kh, kw)
		stride = (sh, sw)
	"""

