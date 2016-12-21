from module import *
from activations import *
from losses import *

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