from variable import VariableBank
from module import module_class_factory, module_types
from layer import layer_class_factory, layer_types


class Network(object):

    def __init__(self):
        self._var_bank = VariableBank()
        self._modules = list() # TODO: upgrade this to DAG
        self._outlet = None

    def top(self, name, *args, **kwargs):
        kw = dict(kwargs)
        trainable = True
        if 'trainable' in kw:
            trainable = kw['trainable']
            del kw['trainable']

        if name in module_types:
            _top_module(self, name, trainable, *args, **kw)
        else:
            _top_layer(self, name, trainable, *args, **kw)


    def _top_module(self, name, *args, **kwargs):
        module_class = module_class_factory(name)
        variables_slot = self._var_bank.initialize(
            name, *args, **kwargs)
        module = module_class(
            variables_slot, *args, **kwargs)
        self._modules.append(module)

    def _top_layer(self, name, *args, **kwargs):
        layer_class = layer_class_factory(name)
        layer = layer_class(*args, **kwargs)
        for description in layer.modules_description:
            module_name, args, kwargs = description
            self.top(module_name, *args, **kwargs)

    def setup_training(self, loss_name, optimizer_name,
                       *args, **kwargs):
        self.stack_module(loss_type, False)
        self._var_bank.set_optimizer(
            optimizer_name, minimize, *args, **kwargs)

    def _forward(self, feed_in, target = None):
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
        self._forward(feed_in)
        self._backprop()
        return self._modules[-1].loss

    def save_checkpoint(self, file_name):
        self._var_bank.save(file_name)

    def load_checkpoint(self, file_name):
        self._var_bank.load(file_name)
