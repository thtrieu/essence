*What I cannot create, I do not understand - Richard Feynman*

### Numpyflow

Is an on-going project that builds deep feed-forward computational graph completely in `numpy` (to be upgrade to DAG). Delegating as much computation as possible to `numpy` built-in routines to optimize for speed. It is supposed to be run entirely on CPU.

Current running modules: matmul, dropout, relu, conv, maxpool2, batchnorm.

TODO: build LSTM, GRU, NTM and train at least one model using each of these modules.

### Test 1: MNIST with depth-2 MLP, relu, dropout

 `numpyflow` code:

```python
mnist = ChainNet()
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

This code presents in `mnist.py` achieves 95.3% accuracy.

### Test 2: LeNet on MNIST with batch normalization

[Writting]

### License
GPL 3.0 (see License in this repo)