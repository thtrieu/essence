from net.network import Network
import numpy as np
import time

batch = 5
inp_dim = 20
hid_dim = 7
out_dim = 3
lr = 1e-2
scale = 1e-3

w1 = np.random.normal(scale = scale, size = (inp_dim, hid_dim))
b1 = np.random.normal(scale = scale, size = hid_dim)
w2 = np.random.normal(scale = scale, size = (hid_dim, out_dim))
b2 = np.random.normal(scale = scale, size = out_dim)

feed = np.random.uniform(-1, 1, size = (batch, inp_dim))
target = np.random.randn(batch, out_dim)
target = np.equal(target, target.max(1, keepdims = True))
target = target.astype(np.float64)

mnist = Network()
mnist.add('dot', w1)
mnist.add('bias', b1)
mnist.add('relu')
mnist.add('dot', w2)
mnist.add('bias', b2)
mnist.add('softmax_crossent')
mnist.set_optimizer('sgd', lr)

s = time.time()
for count in range(30000):
	mnist.train(feed, target)
print time.time() - s

def softmax(x):
	row_max = x.max(1, keepdims = True)
	e_x = np.exp(x - row_max)
	e_sum = e_x.sum(1, keepdims = True)
	return np.divide(e_x, e_sum)

print target, '\n\n\n\n'
print np.around(softmax(mnist._outlet), 1)

