*What I cannot create, I do not understand - Richard Feynman*

### Numpyflow

Is an on-going project that builds auto-differentiable, directed-acyclic computational-graph using `numpy`, with occasional falls back to `C` whenever perfomance is demanded. The interface is inspired by `Tensorflow`. Currenly the project supports CPU only.

Current working layers: fully connected, convolution, dropout, batch normalization, long short term memory

TODO: Augmented memory RNN: namely a Neural Turing Machine (should be cool).

*Motivation:* if there is one algorithm to understand in Deep Learning, that might be Back Propagation. Not chaining derivaties on the paper but actually implement it, experience yourself the vanishing/exploding gradients, witness numerical underflow/overflow in cross-entropy softmax or see the *linear carousel* as one solid line in your code is just wonderful.

### Test 1: MNIST with depth-2 MLP, relu, dropout & train with SGD.

A sample from `mnist-mlp.py`

```python
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
```
This achieves 95.3% accuracy.

### Test 2: LeNet on MNIST with batch normalization & ADAM optimizer

Sample from `lenet-bn.py`

```python
net = Net()
image = net.portal((28, 28, 1))
label = net.portal((10, ))
is_training = net.portal()

# ...

conv1 = net.conv2d(image, k1, pad = (2,2), stride = (1,1))
conv1 = net.batch_norm(
    conv1, net.variable(guass(0., std, (32,))), is_training)
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

This achieves 97% test accuracy.

### Test 3: LSTM on word embeddings for Vietnamese Question classification + Dropout + L2 weight regularize

Only one half of words in the dataset is covered by the embeddings. The many-to-one LSTM is designed to extract only the last relevant outputs for each sentence in batch (they have different lengths and are padded to be equal). After the recurrent is two fully-connected with dropout in between.

A sample from `lstm-embed.py`, training is still in progress.

```python
def lstm_layer(net, embeddings, pos, 
                non_static, lens, hidden_size):
    w = net.lookup(embeddings, pos, trainable = non_static)
    out = net.lstm1(w, lens, hidden_size = hidden_size, forget_bias = 1.5)
    return out

def fully_connected(inp, inp_size, out_size, dropout):
    # ...

lstmed = lstm_layer(embedding, 
    one_hot_sentences, True, lens, 300)
penultimate, regularizer1 = fully_connected(
    lstmed, 300, 128, dropout = True)
penultimate = net.relu(penultimate)
predict, regularizer2 = fully_connected(
    penultimate, 128, nclass)

# Crossent loss combine with weight decay
vanilla_loss = net.softmax_crossent(predict, y)
regularized_loss = net.weighted_loss(
    (vanilla_loss, 1.0), (regularizer1, .2), (regularizer2, .2))
net.optimize(regularized_loss, 'adam', 1e-3)
```

### License
GPL 3.0 (see License in this repo)