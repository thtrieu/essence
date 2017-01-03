from src.net import Net
from src.utils import randn, uniform, guass, read_mnist, accuracy
import numpy as np
from src.net import Net

batch = 8
out_dim = inp_dim = 10
hid_dim = 300
mem_size = 128
vec_size = 30
max_seq_len = 20

start_symbol = np.zeros([batch, 1, inp_dim])
stop_symbol = np.zeros([batch, 1, inp_dim])
start_symbol[:, 0, 0] = np.ones([batch])
stop_symbol[:, 0, 1] = np.ones([batch])

net = Net()
x = net.portal((inp_dim,))
y = net.portal((inp_dim,))
y = net.reshape(y, [-1, inp_dim], over_batch = True)
ntm_out = net.turing(x, out_dim, mem_size, vec_size,
                     hid_dim, shift = 1)

start, end = net.portal(), net.portal()
copy = net.dynamic_slice(
    ntm_out, start = start, end = end, axis = 0)
logits = net.reshape(copy, [-1, inp_dim], over_batch = True)

loss = net.logistic(logits, y)
net.optimize(loss, 'adam', 1e-4)


def generate_random_input(batch, seq_length, inp_dim):
    x = np.random.rand(batch, seq_length, inp_dim).round()
    x[:, :, :2] = np.zeros(x[:, :, :2].shape)
    return x, np.zeros(x.shape) 

net.load_checkpoint('trial7999')

max_seq_len = [5, 10, 15, 20]
epoch_num = [4096, 4096, 8192, 8192]
save_every = 500
for i in range(4):
    max_seq = max_seq_len[i]
    epoch = epoch_num[i]
    for count in xrange(epoch):
        length = np.random.randint(max_seq) + 1
        inp, zeros = generate_random_input(batch, length, inp_dim)
        inp_feed = np.concatenate(
            [start_symbol, inp, stop_symbol, zeros], 1)
        pred, loss = net.train([logits], {
            x: inp_feed, y: inp, 
            start: length + 2, end: 2 * length + 2})

        pred = pred.round()
        acc = np.equal(pred, inp.reshape([-1, inp_dim]))
        acc = acc.astype(np.float64).mean()
        print 'Step {} Loss {} Len/Acc {}/{}'.format(
            count, loss, length, acc)

        if (count + 1) % save_every == 0:
            print 'saving to trial{}'.format(count)
            net.save_checkpoint('trial{}'.format(count))