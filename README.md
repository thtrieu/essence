*What I cannot create, I do not understand - Richard Feynman*

### essence

A directed acyclic computational graph builder, built from scratch on `numpy` and `C`, with auto-differentiation and gradient unit testing.

*Motivation:* if there is one algorithm to understand in Deep Learning, that might be Back Propagation. Not chaining derivaties on paper but the actual implementation of it, see for yourself vanishing/exploding gradients, numerical underflow/overflow and then being able to solve them is just wonderful.

### Demos

1. `mnist-mlp.py`: Depth-2 multi layer perceptron, with ReLU and Dropout; 95.3% on MNIST.

2. `lenet-bn.py`: LeNet with Batch Normalization on first layer, 97% on MNIST.

3. `lstm-embed.py`: LSTM on word embeddings for Vietnamese Question classification + Dropout + L2 weight decay. 85% on test set and 98% on training set (overfit).

4. `turing-copy`: A neural turing machine with LSTM controller. Test result on copy task length 70:

![img](turing.png)


**TODO**: Memory network and GAN, for that I need to improve my speed of `im2col` and `gemm` for `conv` module first.

### License
GPL 3.0 (see License in this repo)