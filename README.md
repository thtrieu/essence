*What I cannot create, I do not understand - Richard Feynman*

### essence

An directed acyclic computational graph builder, built from scratch on `numpy` and `C`, with auto-differentiation and gradient unit testing.

Current working modules: fully connected, convolution, dropout, batch normalization, **LSTM, and Neural Turing Machine, copy task** (see code in four demos below).

TODO: GAN, although I need to improve my implementation of `im2col` and `gemm` for `conv` module.

*Motivation:* if there is one algorithm to understand in Deep Learning, that might be Back Propagation. Not chaining derivaties on paper but the actual implementation of it, see for yourself vanishing/exploding gradients, numerical underflow/overflow and then being able to solve them is just wonderful.

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

### Test 3: LSTM on word embeddings for Vietnamese Question classification + Dropout + L2 weight decay

Only one half of words in the dataset is covered by the embeddings. The many-to-one LSTM is designed to extract only its last relevant outputs for each sentence in batch (they have different lengths and are padded to be equal). After the recurrent is two fully-connected layers with dropout in between.

A sample from `lstm-embed.py`

```python
def lstm_layer(net, embeddings, pos, 
                non_static, lens, hidden_size):
    w = net.lookup(embeddings, pos, trainable = non_static)
    out = net.lstm(w, lens, hidden_size = hidden_size, forget_bias = 1.5)
    # collect the right output since sentences have different length
    out = net.batch_slice(out, lens, axis = 0, shift = -1) 
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

The test accuracy was **85%** while training overfit to **98%**, this tells more regularization is needed, however this is enough for a demonstration as I need to move on for the next thing.

### Test 4: Neural Turing Machine, copy task

A snippet from `turing-copy`

```python
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
```
Test result on length 70:

![img](turing.png)


### License
GPL 3.0 (see License in this repo)