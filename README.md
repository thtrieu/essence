*What I cannot create, I do not understand - Richard Feynman*

### Numpyflow

Is an on-going project that builds deep feed-forward computational graph completely in `numpy` (to be upgrade to DAG). Delegating as much computation as possible to `numpy` built-in routines to optimize for speed. It is supposed to be run entirely on CPU.

Current running modules: matmul, dropout, relu, conv, maxpool2, batchnorm.

TODO: build a syntactic sugar coat for `net.x()`
TODO: build LSTM, GRU, NTM and train at least one model using each of these modules.

### Test 1: MNIST with depth-2 MLP, relu, dropout & train with SGD.

A sample from `mnist-mlp.py`

```python
net = Net()
x = net.x('portal', (inp_dim,))
y = net.x('portal', (out_dim,))

fc1 = net.x('dot', x, w1)
bias = net.x('bias', fc1, b1)
relu = net.x('relu', bias)
dropped = net.x('drop', relu, 0.5)
fc2 = net.x('dot', dropped, w2)
bias = net.x('bias', fc2, b2)
loss = net.x('softmax_crossent', bias, y)
net.optimize(loss, 'sgd', lr)
```
This achieves 95.3% accuracy.

### Test 2: LeNet on MNIST with batch normalization & ADAM optimizer

```python
net = Net()
p = net.x('portal', inp_shape)
is_training = net.x('portal', (1,))
y = net.x('portal', (10, ))

x = net.x('conv', p, k1, (2,2), (1,1))
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
x = net.x('dot', x, w4)
x = net.x('bias', x, b4)
loss = net.x('softmax_crossent', x, y)
net.optimize(loss, 'adam', 1e-3)
```
This achieves 96.7% on MNIST.

### License
GPL 3.0 (see License in this repo)