"""
Code modified from 
https://github.com/iamaaditya/VQA_Demo/blob/master/Visual_Question_Answering_Demo_in_python_notebook.ipynb
Please visit there to get the trained weights :)
"""
import sys, os 

import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn.externals import joblib
from .VGG import VGG_16

VQA_model_file_name     = 'tmp/visualqa/VQA_MODEL.json'
VQA_weights_file_name   = 'tmp/visualqa/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'tmp/visualqa/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'tmp/visualqa/vgg16_weights.h5'


label_encoder_file_name = '/home/thtrieu/numpyflow/tmp/visualqa/FULL_labelencoder_trainval.pkl'
labelencoder = joblib.load(label_encoder_file_name)
all_words = list()
for count in range(1000):
    all_words.append(labelencoder.inverse_transform(count))

def glove_embed(question):
    word_embeddings = spacy.load(
        'en', vectors = 'en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((30, 300))
    for j in range(len(tokens)):
        question_tensor[j, :] = tokens[j].vector
    return question_tensor

def get_image_model(CNN_weights_file_name):
    image_model = VGG_16(CNN_weights_file_name)
    sgd = SGD(
        lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
    image_model.compile(
        optimizer = sgd, loss = 'categorical_crossentropy')
    return image_model

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(
        loss = 'categorical_crossentropy', optimizer = 'rmsprop')
    return vqa_model

def yield_weights(model):
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights): 
            weights = [
                w.astype(np.float64) 
                for w in weights]
            yield(weights)

def vqa_models(VQA_model, VQA_weights):
    class _dummy(object):
        def __init__(self, layers):
            self.layers = layers
    model_vqa = get_VQA_model(
        VQA_model, VQA_weights)
    merged = model_vqa.layers[0]
    lstm3, _ = merged.layers
    vqa_infer = _dummy(model_vqa.layers[1:])
    return lstm3, vqa_infer

def read_image(image_file_name):
    image_features = np.zeros((1, 4096))
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    return im.astype(np.float64).copy(order = 'C')

def to_word(predict):
    label = np.argmax(predict)
    return labelencoder.inverse_transform(label)


vgg16_model = get_image_model(CNN_weights_file_name)
lstm3_model, infer_model = vqa_models(
    VQA_model_file_name, VQA_weights_file_name)