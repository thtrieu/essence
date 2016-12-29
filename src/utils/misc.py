from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import normal as guass
from numpy.random import binomial as binomial
from numpy.random import uniform as uniform
from numpy.random import randn as randn
import numpy as np

def accuracy(predict, truth):
    pred_idx = predict.argmax(1)
    true_idx = truth.argmax(1)
    correct = np.equal(pred_idx, true_idx)
    return correct.mean()

def read_mnist():
    return input_data.read_data_sets('./tmp/data', one_hot = True)

def xavier(shape):
    bound = np.sqrt(6.) / (np.sqrt(sum(shape)))
    return uniform(-bound, bound, shape)

def extract(name, dfault, **kwargs):
    kw = dict(kwargs)
    val = dfault
    if name in kw:
        val = kw[name]
        del kw[name]
    return val, kw

def nxshape(volume, shape):
    n = volume.shape[0]
    return tuple([n] + list(shape))

def parse_for(class_type, *args):
    dep_args = list(); 
    args = list(args)
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if type(arg) is class_type:
            dep_args.append(arg)
            del args[idx]
            idx -= 1
        idx += 1
    return dep_args, args