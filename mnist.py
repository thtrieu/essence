from src.network import ChainNet
from src.utils import randn, uniform, guass, read_mnist
import numpy as np
import time

inp_dim = 784; hid_dim = 128; out_dim = 10
std = 1e-3; lr = 1e-2; batch = 128

w1 = guass(0., std, (inp_dim, hid_dim))
b1 = guass(0., std, hid_dim)
w2 = guass(0., std, (hid_dim, out_dim))
b2 = guass(0., std, out_dim)

mnist = ChainNet()
mnist.add('portal', (inp_dim,))
mnist.add('dot', w1)
mnist.add('bias', b1)
mnist.add('relu')
mnist.add('drop', 0.5)
mnist.add('dot', w2)
mnist.add('bias', b2)
mnist.add('softmax_crossent')
mnist.set_optimizer('sgd', lr)

mnist_data = read_mnist()
s = time.time()
for count in range(30):
	batch_num = int(mnist_data.train.num_examples/batch)
	for i in range(batch_num):
		feed, target = mnist_data.train.next_batch(batch)
		loss = mnist.train(feed, target)
	print 'Epoch {} loss {}'.format(count, loss)
	
print('Total time elapsed: {}'.format(time.time() - s))

mnist.forward(mnist_data.test.images, mnist_data.test.labels)
true_labels = mnist_data.test.labels.argmax(1)
pred_labels = mnist._outlet.argmax(1)
accuracy = np.equal(true_labels, pred_labels).mean()
print accuracy

