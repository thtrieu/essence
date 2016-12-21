import numpy as np
from module import Module

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