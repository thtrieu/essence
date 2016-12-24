from src.net import Net
from src.utils import randn, uniform, guass, read_mnist
import numpy as np
import time

inp_dim = 784; hid_dim = 128; out_dim = 10
std = 1e-3; lr = 1e-2; batch = 128

w1 = guass(0., std, (inp_dim, hid_dim))
b1 = guass(0., std, hid_dim)
w2 = guass(0., std, (hid_dim, out_dim))
b2 = guass(0., std, out_dim)

net = Net()
x = net.x('portal', (inp_dim,))
fc1 = net.x('dot', x, w1)
bias = net.x('bias', fc1, b1)
relu = net.x('relu', bias)
dropped = net.x('drop', relu, 0.5)
fc2 = net.x('dot', dropped, w2)
bias = net.x('bias', fc2, b2)
y = net.x('portal', (out_dim,))
loss = net.x('softmax_crossent', bias, y)
net.optimize(loss, 'sgd', lr)

mnist_data = read_mnist()
s = time.time()
for count in range(30):
	batch_num = int(mnist_data.train.num_examples/batch)
	for i in range(batch_num):
		feed, target = mnist_data.train.next_batch(batch)
		loss = net.train({x: feed, y: target})
	print 'Epoch {} loss {}'.format(count, loss)
	
print('Total time elapsed: {}'.format(time.time() - s))

bias_out = net.forward([bias], {x:mnist_data.test.images})[0]
true_labels = mnist_data.test.labels.argmax(1)
pred_labels = bias_out.argmax(1)
accuracy = np.equal(true_labels, pred_labels).mean()
print accuracy

