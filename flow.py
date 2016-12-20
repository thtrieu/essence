from net.network import Network
import numpy as np
import time

w1 = np.random.uniform(size = (784, 100))
b1 = np.zeros(100)

w2 = np.random.uniform(size = (100, 10))
b2 = np.zeros(10)

mnist = Network()
mnist.top('dot', w1)
mnist.top('bias', b = b1)
mnist.top('drop', keep_prob = .5)
mnist.put('full', w2, b2, trainable = False)
mnist.top('softmax')
mnist.top('crossent')

feed = np.random.randn(128, 784)
target = np.random.randn(128, 10)



# def softmax(x):
# 	row_max = x.max(1, keepdims = True)
# 	e_x = np.exp(x - row_max)
# 	e_sum = e_x.sum(1, keepdims = True)
# 	return np.divide(e_x, e_sum)

# outlet = softmax((feed.dot(w1)+b1).dot(w2)+b2)
# mnist.forward(feed, target)

# print np.array_equal(mnist._outlet, outlet)