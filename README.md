*What I cannot create, I do not understand - Richard Feynman*

### essence

A directed acyclic computational graph builder, built from scratch on `numpy` and `C`, with auto-differentiation supported.

### Demos

- `mnist-mlp.py`: Depth-2 multi layer perceptron, with ReLU and Dropout; 95.3% on MNIST.

- `lenet-bn.py`: LeNet with Batch Normalization on first layer, 97% on MNIST.

- `lstm-embed.py`: LSTM on word embeddings for Vietnamese Question classification + Dropout + L2 weight decay. 85% on test set and 98% on training set (overfit).

- `turing-copy.py`: A neural turing machine with LSTM controller. Test result on copy task length 70:

![img](turing.png)

- `visual-answer.py`. Visual question answering with pretrained weight from VGG16 and an stack of 3 basic LSTMs, on Glove word2vec.

<p align="center"> <img src="test.jpg"/> </p>

```
Question: What is the animal in the picture?
Thinking ...
Answer:
92.73 %  cat
05.18 %  dog
01.03 %  bear
00.55 %  teddy bear
00.21 %  bird
```


**TODO**: Memory network and GAN, for that I need to improve my speed of `im2col` and `gemm` for `conv` module first.

### License
GPL 3.0 (see License in this repo)