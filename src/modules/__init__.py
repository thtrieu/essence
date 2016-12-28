from module import portal
from activations import *
from losses import *
from signatures import *
from common import *
from variable import *

_module_class_factory = dict({
	'portal': portal,
	'conv': conv,
	'maxpool2': maxpool2x2,
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
	'batchnorm': batch_norm,
	'variable': variable,
	'constant': constant
})

def module_class_factory(name):
	assert name in _module_class_factory, \
	'Module {} not implemented'.format(name)
	return _module_class_factory[name]