import numpy as np

class Initializer(object):
    def __init__(self, *args, **kwargs):
        self._val_grad_pairs = dict()

    @property
    def val_grad_pairs(self):
        return self._val_grad_pairs
    

class SingleVariableModule(Initializer):
    def __init__(self, *args, **kwargs):
        args = list(args)
        kwargs = dict(kwargs)
        assert len(args) + len(kwargs) == 1,\
        'Single Variable module must be '\
        'initialized with exactly one variable'

        if len(args): val = args[0]
        else: val = kwargs.values()[0]
        self._val_grad_pairs = dict({
            self._var_name : dict({
                'val': val,
                'grad': None
        })})

class AddBiasModuleInitializer(SingleVariableModule):
    _var_name = 'b'
    def _check_val(self, val):
        shape = val.shape
        assert len(shape) == 1, \
        'Add Bias initialization does not '
        'accept values other than vector'

class MatmulModuleInitializer(SingleVariableModule):
    _var_name = 'w'
    def _check_val(self, val):
        shape = val.shape
        assert len(shape) == 2, \
        'Matmul initialization does not '\
        'accept values other than 2D matrix'

_initializer_class_factory = dict({
    'dot': MatmulModuleInitializer,
    'bias': AddBiasModuleInitializer
})

def initializer_class_factory(name):
    return _initializer_class_factory.get(name, Initializer)
