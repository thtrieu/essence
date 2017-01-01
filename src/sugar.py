
def __getitem__(self, item):
    return self._x('slice', item[0], item[1:])

def turing(self, x, out_size, memory_size, 
            vec_size, controller_size, shift = 1):
    return self._x('turing', x, out_size, memory_size, 
                    vec_size, controller_size, shift)

def batch_slice(self, x, position, axis, shift = 0):
    return self._x('batch_slice', x, position, axis, shift)

def portal(self, shape = (0,)):
    return self._x('portal', shape)

def variable(self, val, trainable = True):
    return self._x('variable', val, trainable)

def lookup(self, val, pos, trainable = True):
    return self._x('lookup', val, pos, trainable)

def constant(self, val):
    return self._x('const', val)

def conv2d(self, x, kernel, pad = (0, 0), stride = (1, 1)):
    return self._x('conv', x, kernel, pad, stride)

def maxpool2(self, x):
    return self._x('maxpool2', x)

def lstm(self, x, lens, hidden_size, forget_bias = None):
    return self._x('lstm', x, lens, hidden_size, forget_bias)

def matmul(self, x, w):
    return self._x('dot', x, w)

def relu(self, x):
    return self._x('relu', x)

def plus_b(self, x, b):
    return self._x('bias', x, b)

def softmax_crossent(self, x, t):
    return self._x('softmax_crossent', x, t)

def logistic(self, x, t):
    return self._x('logistic', x, t)

def batch_norm(self, x, gamma, is_training, momentum = .9):
    return self._x('batchnorm', x, gamma, is_training, momentum)

def reshape(self, x, new_shape, over_batch):
    return self._x('reshape', x, new_shape, over_batch)

def dropout(self, x, keep_prob):
    return self._x('drop', x, keep_prob)

def sigmoid(self, x):
    return self._x('sigmoid', x)

def crossent(self, x, t):
    return self._x('crossent', x, t)

def softmax(self, x):
    return self._x('softmax', x)

def l2(self, x, t):
    return self._x('l2', x, t)

def l2_regularize(self, w, center = 0.):
    return self._x('l2', w, center)

def weighted_loss(self, *pairs):
    losses, weights = zip(*pairs)
    return self._x('weighted_loss', weights, *losses)