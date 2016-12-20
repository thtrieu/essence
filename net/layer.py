import numpy as np

def _pack_description(name, *var_list, **var_dict):
    var_list = list(var_list)
    var_dict = dict(var_dict)
    return name, var_list, var_dict

class Layer(object):
    """Sugar coated set of modules"""
    def __init__(self, *args, **kwargs):
        self._modules_description = list()
        self._construct(*args, **kwargs)

    @property
    def modules_description(self):
        return self._modules_description

    def _construct(self, *args, **kwargs): pass

class FullyConnectedLayer(Layer):
    def _construct(self, w, b):
        self._modules_description = list([
            _pack_description('dot', w),
            _pack_description('bias', b)
        ])

"""
Layer factory
"""

_layer_class_factory = dict({
    'full': FullyConnectedLayer
})

layer_types = _layer_class_factory.keys()

def layer_class_factory(name):
    assert name in _layer_class_factory, \
    'Layer {} not implemented'.format(name)
    return _layer_class_factory[name]
