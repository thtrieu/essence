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

class Activate(Module):
	"""
	Modules whose output participates backprop
	"""
	def _setup(self, *args, **kwargs):
		self.activation = None

	def forward(self, x):
		self.transform(x)
		return self.activation

class sigmoid(Activate):
	def transform(self, x):
		self.activation = 1. / (1. + np.exp(-x))

	def backward(self, grad):
		a = self.activation
		p = np.multiply(a, 1. - a)
		return np.multiply(grad, p)

class linear(Activate):
	def transform(self, x):
		self.activation = x

	def backward(self, grad):
		return grad

class relu(Activate):
	def transform(self, x):
		self.activation = x * (x > 0.)

	def backward(self, grad):
		p = self.activation > 0.
		return np.multiply(grad, p)

class softmax(Activate):
	def transform(self, x):
		row_max = x.max(1, keepdims = True)
		e_x = np.exp(x - row_max)
		e_sum = e_x.sum(1, keepdims = True)
		self.activation = np.divide(e_x, e_sum)

	def backward(self, grad):
		a = self.activation
		m = np.multiply(grad, a)
		g = grad - m.sum(1, keepdims = True)
		return np.multiply(g, a)

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

class Loss(Module):
	def _setup(self, *args, **kwargs):
		self._loss = None

	def set_target(self, target):
		self._t = target

	def forward(self, x):
		self._cal_loss(x)
		return x

	@property
	def loss(self):
		return self._loss

class softmax_crossent(Loss):
	def _cal_loss(self, x):
		row_max = x.max(1, keepdims = True)
		e_x = np.exp(x - row_max)
		e_sum = e_x.sum(1, keepdims = True)
		self._softed = np.divide(e_x, e_sum)
		crossed = - np.multiply(self._t, np.log(self._softed))
		self._loss = crossed.sum(1).mean()

	def backward(self, grad):
		return grad * (self._softed - self._t)

class crossent(Loss):
	def _cal_loss(self, x):
		self._x = x
		crossed = - np.multiply(self._t, np.log(x))
		self._loss = crossed.sum(1).mean()

	def backward(self, grad):
		p = - np.divide(self._t, self._x + 1e-20)
		return grad * p

class l2(Loss):
	def _cal_loss(self, x):
		self._diff = x - self._t
		self._loss = np.pow(self._diff, 2)

	def backward(self, grad):
		return grad * self._d

"""
Module Class Factory
"""

_module_class_factory = dict({
	'reshape': reshape,
	'sigmoid': sigmoid,
	'softmax': softmax,
	'drop': drop,
	'linear': linear,
	'relu': relu,
	'bias': add_biases,
	'dot': matmul,
	'crossent': crossent,
	'softmax_crossent': softmax_crossent,
	'l2': l2,
})

def module_class_factory(name):
	assert name in _module_class_factory, \
	'Module {} not implemented'.format(name)
	return _module_class_factory[name]