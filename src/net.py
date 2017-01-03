from server import ParameterServer
from graph import DAG
import sugar

class Net(object):
    # Sugar syntax
    def __init__(self):
        self._server = ParameterServer()
        self._finalized = False
        self._dagraph = DAG()
        coated = sugar.__dict__
        for name, fun in coated.iteritems():
            if callable(fun): setattr(Net, name, fun)
 
    # To be sugar coated
    def _x(self, name, *args):
        assert not self._finalized, \
        'Graph is finalized by setting an optimizer'
        return self._dagraph.register_for_slot(
            name, self._server, *args)

    def optimize(self, ptr, optimizer,
                 *args, **kwargs):
        self._finalized = True
        self._dagraph.set_closure(ptr)
        self._server.set_optimizer(
            optimizer, *args, **kwargs)

    def forward(self, fetches, feed):
        return self._dagraph.forward(fetches, feed)

    def train(self, fetches, feed):
        val = self._dagraph.forward_to_leaf(fetches, feed)
        self._dagraph.backward()
        self._server.apply_optimizer()
        return val

    def save_checkpoint(self, file_name):
        self._server.save(file_name)

    def load_checkpoint(self, file_name):
        self._server.load(file_name)
