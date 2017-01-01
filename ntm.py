from src.net import Net
from src.utils import randn, uniform, guass, read_mnist, accuracy
import numpy as np
from src.net import Net

batch = 64; epoch = 4000
out_dim = inp_dim = 10
hid_dim = 128;
seq_length = 10;
mem_size = 128; 
vec_size = 20

start_symbol = np.zeros([batch, 1, inp_dim])
stop_symbol = np.zeros([batch, 1, inp_dim])
start_symbol[:, 0, 0] = np.ones([batch])
stop_symbol[:, 0, 1] = np.ones([batch])

net = Net()
# input feed: start + inp + stop + zeros
x = net.portal((seq_length * 2 + 2, inp_dim))
y = net.portal((seq_length, inp_dim))
y = net.reshape(y, [-1, inp_dim], over_batch = True)
ntm_out = net.turing(x, out_dim, mem_size, vec_size,
                     hid_dim, shift = 1)
copy = net[ntm_out, -seq_length:, :]
logits = net.reshape(copy, [-1, inp_dim], over_batch = True)
loss = net.logistic(logits, y)
net.optimize(loss, 'rmsprop', 1e-3)


def generate_random_input(batch, seq_length, inp_dim):
    x = np.random.rand(batch, seq_length, inp_dim).round()
    x[:, :, :2] = np.zeros(x[:, :, :2].shape)
    return x, np.zeros(x.shape)

save_every = 150
for count in xrange(epoch):
    inp, zeros = generate_random_input(batch, seq_length, inp_dim)
    inp_feed = np.concatenate(
        [start_symbol, inp, stop_symbol, zeros], 1)
    pred, loss = net.train([logits], {
        x: inp_feed, y: inp })

    pred = pred.round()
    acc = np.equal(pred, inp.reshape([-1, inp_dim]))
    acc = acc.astype(np.float32).mean()
    print 'Step {} Loss {} Acc {}'.format(count, loss, acc)

    if (count + 1) % save_every == 0:
        net.save_checkpoint('trial{}'.format(count))
        net.load_checkpoint('trial{}'.format(count))