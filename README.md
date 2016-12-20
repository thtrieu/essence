* #table of content
{:toc}

*What I cannot create, I do not understand - Richard Feynman*

### Numpyflow

Is an on-going project that builds deep feed-forward computational graph completely in `numpy`. Delegating as much computation as possible to `numpy` built-in routines to gain a little more spare time.

Goal: finish building CNN, batch normalization, LSTM, GRU, NTM and train at least one model using each of these modules.

### Test 1: speed against `tensorflow`

(Unfair comparision ahead)

```python
batch = 128
inp_dim = 784
hid_dim = 64
out_dim = 10
epoch = 300
lr = 1e-2
scale = 1e-3

w1 = np.random.normal(scale = scale, size = (inp_dim, hid_dim))
b1 = np.random.normal(scale = scale, size = hid_dim)
w2 = np.random.normal(scale = scale, size = (hid_dim, out_dim))
b2 = np.random.normal(scale = scale, size = out_dim)

feed = np.random.uniform(-1, 1, size = (batch, inp_dim))
target = np.random.randn(batch, out_dim)
target = np.equal(target, target.max(1, keepdims = True))
target = target.astype(np.float64)
```

`Tensorflow`'s code:

```python
import tensorflow as tf

w1 = tf.Variable(w1, dtype = tf.float64)
b1 = tf.Variable(b1, dtype = tf.float64)
w2 = tf.Variable(w2, dtype = tf.float64)
b2 = tf.Variable(b2, dtype = tf.float64)
p = tf.placeholder(tf.float64, (batch, inp_dim))
t = tf.placeholder(tf.float64, (batch, out_dim))

x = tf.nn.xw_plus_b(p, w1, b1)
x = tf.nn.relu(x)
x = tf.nn.xw_plus_b(x, w2, b2)
crossent = tf.nn.softmax_cross_entropy_with_logits(x, t)
loss_op = tf.reduce_mean(crossent)

optimizer = tf.train.GradientDescentOptimizer(lr)
gradients = optimizer.compute_gradients(loss_op)
train_op = optimizer.apply_gradients(gradients)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
s = time.time()
for count in range(epoch):
	_, loss = sess.run([train_op, loss_op], feed_dict = {
			p : feed,
			t : target
		})
print time.time() - s
```

```
1.00772094727
```

Equivalent `numpyflow` code:

```python
mnist = Network()
mnist.add('dot', w1)
mnist.add('bias', b1)
mnist.add('relu')
mnist.add('dot', w2)
mnist.add('bias', b2)
mnist.add('softmax_crossent')
mnist.set_optimizer('sgd', lr)

s = time.time()
for count in range(epoch):
	mnist.train(feed, target)
print time.time() - s
```

```
0.620244026184
```

### Test 2: MNIST classification

On MNIST dataset, `mnist.py` achieves 95.3% accuracy.

### License
GPL 3.0 (see License in this repo)