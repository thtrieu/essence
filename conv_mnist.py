from src.network import Network
from src.utils import randn, uniform, guass, read_mnist
import numpy as np
import time

inp_dim = 28; out_dim = 10
std = 1.; lr = 1e-3
inp_shape = (inp_dim, inp_dim, 1)

k1 = guass(0., std, (5, 5, 1, 32))
b1 = guass(0., std, (32,))
k2 = guass(0., std, (5, 5, 32, 64))
b2 = guass(0., std, (64,))
w3 = guass(0., std, (7 * 7 * 64, 1024))
b3 = guass(0., std, (1024,))
w4 = guass(0., std, (1024, 10))
b4 = guass(0., std, (10,))

# k1 = .05 * np.ones((5, 5, 1, 32))
# b1 = -.1 * np.ones((32,))
# k2 = .15 * np.ones((5, 5, 32, 64))
# b2 = -.2 * np.ones((64,))
# w3 = .25 * np.ones((7 * 7 * 64, 1024))
# b3 = -.3 * np.ones((1024,))
# w4 = .35 * np.ones((1024, 10))
# b4 = -.4 * np.ones((10,))

mnist = Network()
mnist.add('portal', inp_shape)
mnist.add('conv', k1, pad = (2,2), stride = (1,1))
mnist.add('bias', b1)
mnist.add('relu')
mnist.add('maxpool2')
mnist.add('conv', k2, pad = (2,2), stride = (1,1))
mnist.add('bias', b2)
mnist.add('relu')
mnist.add('maxpool2')
mnist.add('reshape', (7 * 7 * 64,))
mnist.add('dot', w3)
mnist.add('bias', b3)
mnist.add('relu')
#mnist.add('drop', .75)
mnist.add('dot', w4)
mnist.add('bias', b4)
mnist.add('softmax_crossent')
mnist.set_optimizer('adam', 1e-3)
# mnist.set_saver('mnist', batch * 10)

mnist_data = read_mnist()

def _accuracy():
	mnist.forward(mnist_data.test.images.reshape((-1,28,28,1)), mnist_data.test.labels)
	true_labels = mnist_data.test.labels.argmax(1)
	pred_labels = mnist._outlet.argmax(1)
	accuracy = np.equal(true_labels, pred_labels).mean()
	print 'Trained, accuracy', accuracy

batch = 128
s = time.time()
for count in range(5):
	batch_num = int(mnist_data.train.num_examples/batch)
	for i in range(batch_num):
		feed, target = mnist_data.train.next_batch(batch)
		feed = feed.reshape(batch, 28, 28, 1).astype(np.float32)
		target = target.astype(np.float32)
		loss = mnist.train(feed, target)

		true_labels = target.argmax(1)
		pred_labels = mnist._outlet.argmax(1)
		accuracy = np.equal(true_labels, pred_labels).mean()
		print 'Step {} Loss {} Acc {}'.format(
			i+1 + count*batch_num, loss, accuracy)
		
print time.time() - s
