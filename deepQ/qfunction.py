from src.net import Net
from src.utils.misc import uniform, xavier
import numpy as np

def _fc_with_wb(net, x, w, b, activate):
    linear = net.plus_b(net.matmul(x, w), b)
    return activate(linear)

def _fc(net, x, inp_dim, out_dim, activate):
    w = net.variable(xavier((inp_dim, out_dim)))
    b = net.variable(xavier((out_dim,)))
    out = _fc_with_wb(net, x, w, b, activate)
    return w, b, out

def _MLP2(net, inp_dim, hid_dim, out_dim):
    inp = net.portal((inp_dim,))
    w1, b1, fc1 = _fc(net, inp, inp_dim, hid_dim, net.relu)
    w2, b2, fc2 = _fc(net, fc1, hid_dim, out_dim, lambda x: x)
    return inp, [w1, b1], [w2, b2], fc2

class Qfunction(object):
    def __init__(self, is_target, 
                 inp_dim, hid_dim, out_dim):
        self._net = Net()
        self._inp, fc1_param, fc2_param, self._out = \
            _MLP2(self._net, inp_dim, hid_dim, out_dim)
        self._vars = fc1_param + fc2_param
        if is_target:
            self._feed = list()
            self._assign_ops = list()
            self._build_assign(fc1_param, (inp_dim, hid_dim))
            self._build_assign(fc2_param, (hid_dim, out_dim))
        elif not is_target:
            self._build_loss()
    
    def forward(self, inp_feed):
        out = self._net.forward(
            [self._out],
            {self._inp: inp_feed})[0]
        return out[0, 0]
    
    def train(self, x_feed, y_feed):
        loss, = self._net.train([], {
            self._inp: x_feed,
            self._y: y_feed })
        return loss

    def _build_loss(self):
        self._y = self._net.portal((1,))
        self._loss = self._net.l2(self._out, self._y)
        self._net.optimize(self._loss, 'adam', 1e-3)
     
    def _build_assign(self, fc_weights, shape):
        pw, pb = (
            self._net.portal(shape),
            self._net.portal(shape[1:2]))
        self._feed += [pw, pb]
        self._assign_ops += [
            self._net.assign(fc_weights[0], pw),
            self._net.assign(fc_weights[1], pb)]
    
    def yield_params_values(self):
        returns = list()
        for var in self._vars:
            returns.append(
                self._net.forward([var], {})[0])
        return returns
    
    def assign(self, values):
        feed_dict = dict(zip(self._feed, values))
        self._net.forward(self._assign_ops, feed_dict)
    
    def save(self, file_name):
        self._net.save_checkpoint(file_name)
    
    def load(self, file_name):
        self._net.load_checkpoint(file_name)