from variable import VariableBank
from modules.module import module_class_factory

class Network(object):
    def __init__(self):
        self._var_bank = VariableBank()
        self._modules = list() # TODO: upgrade this to DAG
        self._outlet = None

    def add(self, name, *args, **kwargs):
        variables_slot = self._var_bank.issue(
            name, *args, **kwargs)
        module = module_class_factory(name)(
            variables_slot, *args, **kwargs)
        self._modules.append(module)

    def set_optimizer(self, optimizer_name,
                      *args, **kwargs):
        self._var_bank.set_optimizer(
            optimizer_name, *args, **kwargs)

    def _forward(self, feed_in, target):
        if target is not None: 
            loss_module = self._modules[-1]
            loss_module.set_target(target)

        self._outlet = feed_in
        for module in self._modules:
            self._outlet = module.forward(self._outlet)

    def _backprop(self, grad = 1.0):
        for module in self._modules[::-1]:
            grad = module.backward(grad)
        self._var_bank.apply_optimizer()

    def forward(self, feed_in, target = None):
        self._forward(feed_in, target)
        return self._outlet

    def train(self, feed_in, feed_out):
        self._forward(feed_in, feed_out)
        self._backprop()
        return self._modules[-1].loss

    def save_checkpoint(self, file_name):
        self._var_bank.save(file_name)

    def load_checkpoint(self, file_name):
        self._var_bank.load(file_name)
