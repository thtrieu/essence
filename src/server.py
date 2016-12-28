from optimizer import optimizer_factory
from slot import VariableSlot, MovingVariableSlot

class ParameterServer(object):
    """
    A server of variables, consists of slots
    each slot contains variables of one module, thus
    each slot can have zero, one or more variables.
    """
    
    def __init__(self):
        self._slots = list()

    def issue_movingvar_slot(self, shape, momen):
        slot = MovingVariableSlot(shape, momen)
        self._slots.append(slot)
        return slot

    def issue_var_slot(self, val, trainable):
        slot = VariableSlot(val, trainable)
        self._slots.append(slot)
        return slot
    
    def set_optimizer(self, name, *args, **kwargs):
        optimizer = optimizer_factory(
            name, *args, **kwargs)
        self._optimizer = optimizer

    def apply_optimizer(self):
        for slot in self._slots:
            self._optimizer.apply(slot)
    
    def save(file_name):
        pass
        
    def load(file_name):
        pass