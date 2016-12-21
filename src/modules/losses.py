import numpy as np
from module import Module

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