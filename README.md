*What I cannot create, I do not understand - Richard Feynman*

### essence

A directed acyclic computational graph builder, built from scratch on `numpy` and `C`, including auto-differentiation.

This was not *just another deep learning library*, its minimal code base was supposed to demonstrate how to:

* Build neural net modules.
* Put the modules together.
* Efficiently compute gradients from this design.

### Demos

- `mnist-mlp.py`: Depth-2 multi layer perceptron, with ReLU and Dropout; 95.3% on MNIST.

- `lenet-bn.py`: LeNet with Batch Normalization on first layer, 97% on MNIST.

- `lstm-embed.py`: LSTM on word embeddings for Vietnamese Question classification + Dropout + L2 weight decay. 85% on test set and 98% on training set (overfit).

- `turing-copy.py`: A neural turing machine with LSTM controller. Test result on copy task length 70:

![img](turing.png)

- `visual-answer.py`. Visual question answering with *pretrained* weight from VGG16 and a stack of 3 basic LSTMs, on Glove word2vec.

<p align="center"> <img src="test.jpg"/> </p>

```
Q: What is the animal in the picture?      . A: cat
Q: Is there any person in the picture?     . A: no
Q: What is the cat doing?                  . A: sitting
Q: Where is the cat sitting on?            . A: floor
Q: What is the cat color?                  . A: white
Q: Is the cat smiling?                     . A: yes
```

- `dqn-cartpole.py`: A classic solved with DQN, with experience replay and target network ofcourse. (Illustration below is one-take)

<p align="center"> <img src="demo_cartpole.gif"/> </p>

**TODO**: Memory network and GAN, for that I need to improve my speed of `im2col` and `gemm` for `conv` module first.

### License
GPL 3.0 (see License in this repo)