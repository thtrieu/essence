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
image = net.portal((28, 28, 1))
label = net.portal((10, ))
is_training = net.portal()

conv1 = net.conv2d(image, k1, pad = (2,2), stride = (1,1))
conv1 = net.batch_norm(
    conv1, is_training, 
    gamma = guass(0., std, ()), 
    moving_mean = np.zeros((32,)), 
    moving_var = np.zeros((32,))
)
conv1 = net.plus_b(conv1, b1)
conv1 = net.relu(conv1)
pool1 = net.maxpool2(conv1)

conv2 = net.conv2d(pool1, k2, (2,2), (1,1))
conv2 = net.plus_b(conv2, b2)
conv2 = net.relu(conv2)
pool2 = net.maxpool2(conv2)

flat = net.reshape(pool2, (7 * 7 * 64,))
fc1 = net.plus_b(net.matmul(flat, w3), b3)
fc1 = net.relu(fc1)

fc2 = net.plus_b(net.matmul(fc1, w4), b4)
loss = net.softmax_crossent(fc2, label)
net.optimize(loss, 'adam', 1e-3)

mnist_data = read_mnist()

batch = 128
for count in range(5):
    batch_num = int(mnist_data.train.num_examples/batch)
    for i in range(batch_num):
        feed, target = mnist_data.train.next_batch(batch)
        feed = feed.reshape(batch, 28, 28, 1).astype(np.float32)
        target = target.astype(np.float32)

        pred, cost = net.train([fc2], {
            image: feed, 
            label: target, 
            is_training: True})

        predict = pred.argmax(1)
        truth = target.argmax(1)
        accuracy = np.equal(predict, truth).mean()

        print 'Step {} Loss {} Accuracy {}'.format(
            i+1 + count*batch_num, cost, accuracy)


predict = net.forward([fc2], {
    x : mnist_data.test.images.reshape((-1,28,28,1))
    })[0]
true_labels = mnist_data.test.labels.argmax(1)
pred_labels = predict.argmax(1)
accuracy = np.equal(true_labels, pred_labels).mean()
print 'Accuracy on test set:', accuracy
