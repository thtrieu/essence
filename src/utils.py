from numpy.random import normal as guass
from numpy.random import binomial as binomial
from numpy.random import uniform as uniform
from numpy.random import randn as randn
from tensorflow.examples.tutorials.mnist import input_data

def read_mnist():
    return input_data.read_data_sets('./tmp/data', one_hot = True)

def extract(name, dfault, **kwargs):
    kw = dict(kwargs)
    val = dfault
    if name in kw:
        val = kw[name]
        del kw[name]
    return val, kw