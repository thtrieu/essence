import numpy as np
import re
import itertools
from collections import Counter
import cPickle as pickle
import os

def _clean_str_vn(string):
    string = re.sub(r"[~`@#$%^&*+-]", " ", string)
    # This transforms acronyms into sharp(#)
    string = re.sub('\s[A-Za-z]\s\.', '#', ' '+string)
    string = re.sub('#{2,}', '#', string)
    # Normal regularization
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\.{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def _load_data_and_labels():
    # Load data from files
    folder_prefix = 'tmp/TRECvn/'
    x_train = list(open(folder_prefix+"train").readlines())
    x_test = list(open(folder_prefix+"test").readlines())
    test_size = len(x_test)
    x_text = x_train + x_test

    # Split
    x_text = [_clean_str_vn(sent) for sent in x_text]
    y = [s.split(' ')[0].split(':')[0] for s in x_text]
    x_text = [s.split(" ")[1:] for s in x_text]

    # Generate labels
    all_label = dict()
    for label in y:
        if not label in all_label: 
            all_label[label] = len(all_label) + 1
    one_hot = np.identity(len(all_label))
    y = [one_hot[ all_label[label]-1] for label in y]
    return [x_text, y, test_size]

def _load_trained_vecs(vocabulary):
    bin_prefix = folder_prefix = 'tmp/TRECvn/'
    if not os.path.exists(folder_prefix + 'trained_vecs.PICKLE'):
        binfile = 'vectors-phrase.bin.vn'
        trained_vecs = _load_bin_vec(bin_prefix + binfile, vocabulary)
        with open(folder_prefix + 'trained_vecs.PICKLE', 'wb') as f:
            pickle.dump([trained_vecs], f, protocol=-1)
    else:
        with open(folder_prefix + 'trained_vecs.PICKLE', 'rb') as f:
            trained_vecs = pickle.load(f)[0]
    return trained_vecs

def _pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence. Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def _build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # vocabulary_inv=['<PAD/>', 'the', ....]
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def _build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def _load_data():
    """
    Loads and preprocessed data
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, test_size = _load_data_and_labels()
    sentences_padded = _pad_sentences(sentences)
    vocabulary, vocabulary_inv = _build_vocab(sentences_padded)
    x, y = _build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, test_size]

def _load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def _add_unknown_words(word_vecs, vocab, min_df=0, k=300):
    """	    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    count = 0
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
        else:
            count += 1
    return count  

class TRECvn(object):
    def __init__(self, holdout = 300):
        x_, y_, vocab, vocab_inv, test_size = _load_data()
        trained_vecs = _load_trained_vecs(vocab)
        added = _add_unknown_words(trained_vecs, vocab)
        embeddings = [trained_vecs[p] for p in vocab_inv]
        embeddings = np.array(embeddings, dtype = np.float32)
        self._embeddings = embeddings

        x, x_test = x_[:-test_size], x_[-test_size:]
        y, y_test = y_[:-test_size], y_[-test_size:]
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        if holdout == 0:
            x_train = x_shuffled
            y_train = y_shuffled
            x_dev = x_test
            y_dev = y_test
        else:
            # Split train/hold-out/test set
            x_train, x_dev = x_shuffled[:-holdout], x_shuffled[-holdout:]
            y_train, y_dev = y_shuffled[:-holdout], y_shuffled[-holdout:]

        print("Vocabulary Size: {:d}".format(len(vocab)))
        print("Pre-trained words: {:d}".format(added))
        print("Train/Hold-out/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))

        self._max_len = x_dev.shape[1]
        self._nclass = y_dev.shape[1]
        self._x_train = x_train; self._y_train = y_train
        self._x_test = x_test; self._y_test = y_test
        self._x_dev = x_dev; self._y_dev = y_dev
    
    @property
    def max_len(self):
        return self._max_len

    @property
    def nclass(self):
        return self._nclass

    @property
    def embeddings(self):
        return self._embeddings
        
    def yield_batch(self, batch_size, num_epochs):
        data_size = self._x_train.shape[0]
        num_batches_per_epoch = int(data_size/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = self._x_train[shuffle_indices]
            shuffled_y = self._y_train[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = (batch_num + 1) * batch_size
                if end_index > data_size:
                    end_index = data_size
                    start_index = end_index - batch_size
                yield shuffled_x[start_index: end_index],\
                      shuffled_y[start_index: end_index]

    def yield_test(self):
        return self._x_test, self._y_test
    def yield_dev(self):
        return self._x_dev, self._y_dev