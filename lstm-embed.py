from src.net import Net
from src.utils import randn, uniform, guass, read_mnist, accuracy
import numpy as np
from src.net import Net
from src.utils import TRECvn

def portals(net, max_len, nclass):
    x = net.portal((max_len,))
    y = net.portal((nclass,))
    keep = net.portal()
    real_len = net.portal()
    return x, y, keep, real_len

def lstm_layer(net, embeddings, pos, 
                non_static, lens, hidden_size):
    w = net.lookup(embeddings, pos, trainable = non_static)
    out = net.lstm1(w, lens, hidden_size = hidden_size)
    return out

def fully_connected(inp, inp_size, out_size, center, drop = None):
    w = net.variable(guass(0., .1, (inp_size, out_size)))
    regularizer = net.l2_regularize(w, center)
    b = net.variable(np.ones((out_size,)) * .1)
    mul = net.matmul(inp, w)
    biased = net.plus_b(mul, b)
    if drop is not None:
        biased = net.dropout(biased, drop)
    return biased, regularizer


dat = TRECvn(holdout = 0) # set holdout > 0 for early stopping
max_len, nclass = dat.max_len, dat.nclass
lstm_cell = 300
fc_size = 128

net = Net()
# Set up the portals and recurrent layer
x, y, keep, lens = portals(net, max_len, nclass)
lstmed = lstm_layer(
    net, dat.embeddings, x, 
    True, lens, lstm_cell)

# Setup two fully-connected layer with middle dropout
# And L2 weight decay for both of them.
center = net.portal((1,)) # to be fed by zero.
penultimate, regularizer1 = fully_connected(
    lstmed, lstm_cell, fc_size, center, drop = keep)
penultimate = net.relu(penultimate)
predict, regularizer2 = fully_connected(
    penultimate, fc_size, nclass, center)

# Crossent loss combine with weight decay
vanilla_loss = net.softmax_crossent(predict, y)
regularized_loss = net.weighted_loss(
    (vanilla_loss, 1.0), (regularizer1, .2), (regularizer2, .2))
net.optimize(regularized_loss, 'adam', 1e-4)

# Helper functions
def real_len(x_batch):
    return [np.argmin(s + [0]) for s in x_batch]

# Training
batch = int(64); epoch = int(15); count = int(0)
for sentences, label in dat.yield_batch(batch, epoch):
    pred, loss = net.train([predict], {
        x: sentences, y: label, keep: .75,
        lens: real_len(sentences), center : 0. })
    count += 1
    acc = accuracy(pred, label)
    x_test, y_test = dat.yield_test()
    pred = net.forward([predict], {
        x: x_test, keep: 1., lens: real_len(x_test) })[0]
    acc2 = accuracy(pred, y_test)
    print 'Step {}, Loss {}, Accuracy {}, Acc{}'.format(
        count, loss, acc, acc2)