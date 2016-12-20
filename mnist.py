from src.network import Network
from numpy.random import normal as guass
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('/tmp/data', one_hot = True)

inp_dim = 784; hid_dim = 128; out_dim = 10
std = 1e-3; lr = 1e-4; batch = 128

w1 = guass(0., std, (inp_dim, hid_dim))
b1 = guass(0., std, hid_dim)
w2 = guass(0., std, (hid_dim, out_dim))
b2 = guass(0., std, out_dim)

feed = np.random.uniform(-1., 1., (batch, inp_dim))
target = np.random.randn(batch, out_dim)
target = np.equal(target, target.max(1, keepdims = True))
target = target.astype(np.float64)

mnist = Network()
mnist.add('dot', w1)
mnist.add('bias', b1)
mnist.add('relu')
mnist.add('drop', 0.5)
mnist.add('dot', w2)
mnist.add('bias', b2)
mnist.add('softmax_crossent')
mnist.set_optimizer('sgd', lr)

s = time.time()
for count in range(100):
	batch_num = int(mnist_data.train.num_examples/batch)
	for i in range(batch_num):
		feed, target = mnist_data.train.next_batch(batch)
		loss = mnist.train(feed, target)
	print 'Epoch {} loss {}'.format(count, loss)
print time.time() - s

mnist.forward(mnist_data.test.images, mnist_data.test.labels)
true_labels = mnist_data.test.labels.argmax(1)
pred_labels = mnist._outlet.argmax(1)
accuracy = np.equal(true_labels, pred_labels).mean()
print accuracy

