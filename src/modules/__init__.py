from module import portal, variable, constant, lookup
from activations import *
from losses import *
from signatures import *
from common import *
from shaping import *
from lstm import lstm
from turing import turing

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
	'batchnorm': batch_norm,
	'variable': variable,
	'constant': constant,
	'lookup': lookup,
	'lstm': lstm,
	'crossent': crossent,
	'softmax_crossent': softmax_crossent,
	'l2': l2,
	'weighted_loss': weighted_loss,
	'slice': slices,
	'batch_slice': batch_slice,
	'turing': turing,
	'logistic': logistic,
})

def module_class_factory(name):
	assert name in _module_class_factory, \
	'Module {} not implemented'.format(name)
	return _module_class_factory[name]