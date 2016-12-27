*What I cannot create, I do not understand - Richard Feynman*

### Numpyflow

Is an on-going project that builds auto-differentiable, directed-acyclic computational-graph using `numpy`, with occasional falls back to `C` whenever perfomance is demanded. The interface is inspired by `Tensorflow`. Currenly the project supports CPU only.

Current working layers: fully connected, convolution, dropout, batch normalization. 

TODO: recurrent models: LSTM, GRU, possibly augmented memory RNN such as NTM.

### Test 1: MNIST with depth-2 MLP, relu, dropout & train with SGD.

A sample from `mnist-mlp.py`

```python
net = Net()
image = net.portal((784,))
keep_prob = net.portal()
target = net.portal((10,))

fc1 = net.matmul(image, w1)
bias = net.plus_b(fc1, b1)
relu = net.relu(bias)
dropped = net.dropout(relu, keep_prob)
fc2 = net.matmul(dropped, w2)
bias = net.plus_b(fc2, b2)
loss = net.softmax_crossent(bias, target)

net.optimize(loss, 'sgd', lr)
```
This achieves 95.3% accuracy.

### Test 2: LeNet on MNIST with batch normalization & ADAM optimizer

Sample from `lenet-bn.py`

```python
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
```

This achieves 96.83% test accuracy.

### License
GPL 3.0 (see License in this repo)