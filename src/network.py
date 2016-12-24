from server import ParameterServer
from graph import Chain
from modules import module_class_factory
import time

class Network(object):
    def __init__(self):
        self._server = ParameterServer()
        self._chain = Chain()
        self._outlet = None

    def add(self, name, *args, **kwargs):
        variables_slot = self._server.issue_slot(
            name, *args, **kwargs)
        args, kwargs = variables_slot.hyper_pars

        module = module_class_factory(name)(
            variables_slot, self._chain.current_shape, 
            *args, **kwargs)
        self._chain.add_node(module)

    def set_optimizer(self, optimizer_name,
                      *args, **kwargs):
        self._server.set_optimizer(
            optimizer_name, *args, **kwargs)

    def _forward(self, feed_in, target):
        if target is not None:
            self._chain.leaf.set_target(target)
        self._outlet = feed_in
        for module in self._chain.forth_traverse():
            self._outlet = module.forward(self._outlet)

    def _backprop(self, grad = 1.0):
        for module in self._chain.back_traverse():
            grad = module.backward(grad)
        self._server.apply_optimizer()

    def forward(self, feed_in, target = None):
        self._forward(feed_in, target)
        return self._outlet

    def train(self, feed_in, feed_out):
        self._forward(feed_in, feed_out)
        self._backprop()
        return self._chain.leaf.loss

    def save_checkpoint(self, file_name):
        self._server.save(file_name)

    def load_checkpoint(self, file_name):
        self._server.load(file_name)
