from .gradcheck import GradientChecker

class Module(object):
    def __init__(self, *args): pass

    def forward(self, *args): pass

    def backward(self, grad): pass

    @property
    def out_shape(self):
        return tuple(self._out_shape)
    
    def unit_test(self, x, y):
        GradientChecker.check(self, x, y)

class portal(Module):
    def __init__(self, server, shape):
        self._inp_shape = tuple(shape)
        self._out_shape = tuple(shape)
    def forward(self, x): return x
    def backward(self, grad): return grad

class variable(Module):
    def __init__(self, server, val, trainable):
        self._var = server.issue_var_slot(val, trainable)
        self._out_shape = val.shape
    
    def forward(self):
        return self._var.val
    
    def backward(self, grad):
        self._var.set_grad(grad)

class lookup(Module):
    def __init__(self, server, pos_shape, val, trainable):
        self._var = server.issue_var_slot(val, trainable)
        self._out_shape = (pos_shape[0], val.shape[1])
    
    def forward(self, pos):
        self._pos = pos
        return self._var.lookup(pos)
    
    def backward(self, grad):
        self._var.set_grad((grad, self._pos))

class constant(Module):
    def __init__(self, server, val):
        self._var = server.issue_var_slot(self, val, False)
    
    def forward(self):
        return self._var.val
    
    def backward(self, grad):
        pass