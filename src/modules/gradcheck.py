import numpy as np
from src.utils import uniform, guass, randint

class GradientChecker(object):
    threshold = np.float64(1e-3)
    epsilon = np.float64(1e-3)
    check_round = int(8)
    current = None

    @classmethod
    def tolist(cls, obj):
        if type(obj) not in (list, tuple): obj = [obj]
        return obj

    @classmethod
    def check(cls, module, args, outputs):
        if module == cls.current: return
        print 'Checking', module
        cls.current = module
        outputs = cls.tolist(outputs)
        for out_idx in range(len(outputs)):
            cls.check_against_output(
                module, args, outputs, out_idx)
        cls.current = None

    @classmethod
    def check_against_output(
            cls, module, args, outs, out_idx):
        target = outs[out_idx]
        simulated_target = guass(
            target.mean(), target.std() + 1., target.shape)
        out_grads = [np.zeros(o.shape) for o in outs]
        out_grads[out_idx] = 2 * (target - simulated_target)
        inp_grads = module.backward(*out_grads)
        if inp_grads is None: return

        print '\tChecking w.r.t output #', out_idx
        inp_grads = cls.tolist(inp_grads)
        for inp_idx, inp_grad in enumerate(inp_grads):
            if type(inp_grad) is np.ndarray:
                cls.check_against_input(
                    module, args, inp_idx, inp_grad, 
                    out_idx, simulated_target)

    @classmethod
    def check_against_input(cls, module, args, inp_idx, 
                            grad, out_idx, y):
        print '\t\tChecking input #', inp_idx

        x = args[inp_idx]; relates = list()
        assert x.shape == grad.shape, \
        'Gradient shape mismatch: {} and {}'.format(x.shape, grad.shape)

        for each in range(cls.check_round):
            pick = tuple(randint(dim) for dim in x.shape)
            x1, x2 = x.copy(), x.copy()
            x1[pick] -= cls.epsilon 
            x2[pick] += cls.epsilon

            args[inp_idx] = x1
            y1 = cls.tolist(module.forward(*args))[out_idx]
            L1 = (y1 - y) * (y1 - y)

            args[inp_idx] = x2
            y2 = cls.tolist(module.forward(*args))[out_idx]
            L2 = (y2 - y) * (y2 - y)

            grad_pick = (L2.sum() - L1.sum()) 
            grad_pick /= (2. * cls.epsilon)

            if grad_pick == grad[pick]: 
                print '\t\t', grad_pick, grad[pick]
                continue
            if grad_pick * grad[pick] == 0. and \
                max(abs(grad_pick), abs(grad[pick])) < 1e-15:
                print '\t\tfine'
            relate = abs(grad_pick - grad[pick])
            relate /= max(abs(grad_pick), abs(grad[pick]))
            print '\t\t', grad_pick, grad[pick], relate
            relates.append(relate)

        if len(relates) == 0: return
        relate = np.mean(relates)
        # assert relate < cls.threshold, \
        # 'Gradcheck failed at {}'.format(module)