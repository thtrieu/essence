from src.net import Net
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


net = Net()
p = net.x('portal', inp_shape)
conv1 = net.x('conv', p, k1, (2,2), (1,1))
is_training = net.x('portal', (1,))
x = net.x('batchnorm', x, is_training, guass(0., std, ()),
    np.zeros((32,)), np.zeros((32,)))
x = net.x('bias', x, b1)
x = net.x('relu', x)
x = net.x('maxpool2', x)
x = net.x('conv', x, k2, (2,2), (1,1))
x = net.x('bias', x, b2)
x = net.x('relu', x)
x = net.x('maxpool2', x)
x = net.x('reshape', x, (7 * 7 * 64,))
x = net.x('dot', x, w3)
x = net.x('bias', x, b3)
x = net.x('relu', x)
# = net.x('drop', .75)
x = net.x('dot', x, w4)
x = net.x('bias', x, b4)
y = net.x('portal', (10, ))
loss = net.x('softmax_crossent', x, y)
net.optimize(loss, 'adam', 1e-3)
# mnist.set_saver('mnist', batch * 10)

mnist_data = read_mnist()

def _accuracy():
    b = net.forward([bias], {x : mnist_data.test.images.reshape((-1,28,28,1))})[0]
    true_labels = mnist_data.test.labels.argmax(1)
    pred_labels = b.argmax(1)
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
        loss = net.train({p: feed, y: target, is_training: True})

        x_ = net.forward([x], {p:feed, is_training:True})[0]
        true_labels = target.argmax(1)
        pred_labels = x_.argmax(1)
        accuracy = np.equal(true_labels, pred_labels).mean()
        print 'Step {} Loss {} Acc {}'.format(
            i+1 + count*batch_num, loss, accuracy)
        
print time.time() - s
