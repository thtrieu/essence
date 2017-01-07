import numpy as np
from src.net import Net
from src.utils.visualqa import yield_weights, to_word
from src.utils.visualqa import glove_embed, read_image
from src.utils.visualqa import vgg16_model, lstm3_model, infer_model

'''
Model:

lstm x 3 -----\
               > concatenate -> (dense, tanh) x 3 -> (dense, softmax)
vgg16   ------/

From: https://github.com/iamaaditya/VQA_Demo
'''

vgg16_weights = yield_weights(vgg16_model)
lstm3_weights = yield_weights(lstm3_model)
infer_weights = yield_weights(infer_model)

def build_n_convs(net, inp, n):
    # build n sequential convolutions
    def _build_1_conv(inp):
        kernel, bias = next(vgg16_weights)
        kernel = kernel.transpose([2, 3, 1, 0])
        kernel = kernel.copy(order = 'C')
        kernel, bias = map(
            net.variable, [kernel, bias])
        conved = net.conv2d(
            inp, kernel, pad = (1,1), stride = (1,1))
        return net.relu(net.plus_b(conved, bias))

    conved = inp
    for count in range(n):
        conved = _build_1_conv(conved)
    return net.maxpool2(conved)
    
def build_n_linears(net, inp, n):
    # build n sequential fully connected
    def _build_1_linear(inp):
        weight, bias = next(vgg16_weights)
        weight, bias = map(
            net.variable, [weight, bias])
        lineared = net.matmul(inp, weight)
        return net.relu(net.plus_b(lineared, bias))
        
    lineared = inp
    for count in range(n):
        lineared = _build_1_linear(lineared)
    return lineared

def build_vgg16(net, img):
    conved = img
    for n in [2, 2, 3, 3, 3]:
        conved = build_n_convs(net, conved, n)
    # vgg16_weights are in NCHW order
    tran = net.transpose(conved, [2, 0, 1])
    flat = net.reshape(tran, (7 * 7 * 512,))
    return build_n_linears(net, flat, 2)

def build_lstm3(net, question, real_len):
    # build 3 sequential lstms
    def build_1_lstm(inp, real_len):
        w = next(lstm3_weights)
        gates = [w[i: i + 3] \
            for i in range(0, len(w), 3)]
        tmp = dict()
        # order (keras, essence): (igfo, fiog)
        for gate, k in zip(gates, 'igfo'):
            wx, wh, b = gate
            whx = np.concatenate([wh, wx])
            tmp[k] = (whx, b)
        transfer = list(map(tmp.__getitem__, 'fiog'))
        return net.lstm(
            inp, real_len, 512,
            gate_activation = 'hard_sigmoid',
            transfer = transfer)

    lstmed = question
    for count in range(int(3)):
        lstmed = build_1_lstm(lstmed, real_len)
    last_time_step = net.batch_slice(
        lstmed, real_len, shift = -1)
    return last_time_step

def build_infer(net, inp):
    def _fc(inp, act):
        weight, bias = next(infer_weights)
        weight, bias = map(
            net.variable, [weight, bias])
        lineared = net.matmul(inp, weight)
        return act(net.plus_b(lineared, bias))
    
    fc1 = _fc(inp, net.tanh)
    fc2 = _fc(fc1, net.tanh)
    fc3 = _fc(fc2, net.tanh)
    fc4 = _fc(fc3, net.softmax)
    return fc4

# Build the net.
net = Net()
image = net.portal((224, 224, 3))
question = net.portal((40, 300))
real_len = net.portal((1,))

vgg16_feat = build_vgg16(net, image)
lstm3_feat = build_lstm3(net, question, real_len)
infer_feat = net.concat([lstm3_feat, vgg16_feat])
answer = build_infer(net, infer_feat)

image_feed = read_image('test.jpg')
queries = [
    u"What is the animal in the picture?",
    u"Where is the cat sitting on?",
    u"Is it male or female?",
    u"Is she smiling?",
    u"What is her color?"
]

query_feed = list()
for query in queries:
     query_feed.append(glove_embed(query))
image_feed = [image_feed] * len(queries)
image_feed = np.array(image_feed)
query_feed = np.array(query_feed)

predicts, = net.forward([answer], {
    image: image_feed,
    question : query_feed,
    real_len: [30] * len(queries) 
})

for i, predict in enumerate(predicts):
    print('Q: {:<40}. A: {}'.format(
        queries[i], to_word(predict)))