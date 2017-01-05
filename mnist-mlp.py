from src.net import Net
from src.utils import randn, uniform, guass, read_mnist
import numpy as np
import time

inp_dim = 784; hid_dim = 64; out_dim = 10
std = 1e-3; lr = 1e-2; batch = 128

net = Net()
image = net.portal((784,))
keep_prob = net.portal()
target = net.portal((10,))

w1 = net.variable(guass(0., std, (inp_dim, hid_dim)))
b1 = net.variable(np.ones(hid_dim) * .1)
w2 = net.variable(guass(0., std, (hid_dim, out_dim)))
b2 = net.variable(np.ones(out_dim) * .1)

fc1 = net.matmul(image, w1)
bias = net.plus_b(fc1, b1)
relu = net.relu(bias)
dropped = net.dropout(relu, keep_prob)
fc2 = net.matmul(dropped, w2)
bias = net.plus_b(fc2, b2)
loss = net.softmax_crossent(bias, target)

net.optimize(loss, 'sgd', lr)

mnist_data = read_mnist()
s = time.time()
for count in range(30):
	batch_num = int(mnist_data.train.num_examples/batch)
	for i in range(batch_num):
		img, lab = mnist_data.train.next_batch(batch)
		loss = net.train([], {
			image: img, 
			target: lab,
			keep_prob: .5,
		})[0]
	print('Epoch {} loss {}'.format(count, loss))
	
print('Total time elapsed: {}'.format(time.time() - s))

bias_out = net.forward([bias], {image:mnist_data.test.images})[0]
true_labels = mnist_data.test.labels.argmax(1)
pred_labels = bias_out.argmax(1)
accuracy = np.equal(true_labels, pred_labels).mean()
print('Accuracy on test set', accuracy)

