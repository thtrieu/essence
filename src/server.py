from optimizer import optimizer_factory
from slot import VariableSlot, MovingVariableSlot
import cPickle as pickle

class ParameterServer(object):
    """
    A server of variables, consists of slots
    each slot contains variables of one module, thus
    each slot can have zero, one or more variables.
    """
    
    def __init__(self):
        self._slots = list()
        self._optimizer = None

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
        self._optimizer.finalize_step()
    
    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(
                [self._slots, self._optimizer], f, protocol = -1)
        
    def load(self, file_name):
        with open(file_name, 'rb') as f:
            slots, self._optimizer = pickle.load(f)
        assert len(slots) == len(self._slots)
        for i, slot in enumerate(self._slots):
            slot.load(slots[i])