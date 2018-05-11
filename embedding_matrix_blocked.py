# -*- coding: utf-8 -*-
'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import TimeDistributed, Bidirectional, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, Dropout
from keras.layers import Dense, Input
from keras import backend as K
from keras import optimizers
import pickle

BASE_DIR = ''
LABEL_NB = 9
GLOVE_DIR = BASE_DIR + 'glove.6B/'
CORPUS = BASE_DIR + 'song/all_songs'
TRAIN_FNAME = BASE_DIR + 'tag_cat/train_data'
TEST_FNAME = BASE_DIR + 'tag_cat/test_data'
VALID_FNAME = BASE_DIR + 'tag_cat/validation_data'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 100
MAX_LINE_LENGTH = 10
MAX_BLOCK_LENGTH = 10
MAX_SONG_LENGTH = 6

dropout = 0.5
attention_size=100
n_hidden=50
num_genres=9
learning_rate = .01 # tuned on dev set
training_epochs = 10
max_grad_norm = 1. # tuned on dev set
batch_size=64

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')


def label_matrix(f_table):
    mat = []
    for row in f_table:
        mat_t = [0] * LABEL_NB
        label = [int (i) for i in row[4].split(',')]
        for i in label:
            mat_t[i] = 1
        mat.append(mat_t)
    return mat


def add_file(fname):
    f_t = open(fname, 'rb')
    f = pickle.load(f_t)
    f_t.close()
    texts = []
    if len(f[0][4]) > 0:
        label_mat = label_matrix(f)
    else:
        label_mat = []
    length = len(f)
    for s in f:
        block = []
        for i, l in enumerate(s[5]):
            if len(l[2]) > 0:
                block.append(l[2])
                block_len
            else:
                while len(block) < MAX_BLOCK_LENGTH:
                    block.append('')
            if len(block) == MAX_BLOCK_LENGTH:
                texts.append(block)
                block = []


            # if i == MAX_SONG_LENGTH:
            #     break
            # texts.append(l[2])
        length = len(texts)
        while length < MAX_SONG_LENGTH:
            texts.append("")
            length += 1
            # print(l[2])
    return texts, length, label_mat


corpus_texts, corpus_length, _ = add_file(CORPUS)
train_texts, train_length, train_labels = add_file(TRAIN_FNAME)
print (train_labels[:5])
test_texts, test_length, test_labels = add_file(TEST_FNAME)
valid_texts, valid_length, valid_labels = add_file(VALID_FNAME)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(corpus_texts)


def seq_generation(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    for i in range(len(sequences)):
        size = min(MAX_LINE_LENGTH, len(sequences[i]))
        to_add = MAX_LINE_LENGTH - size
        sequences[i] = sequences[i][:size] + to_add * [MAX_NB_WORDS + 1]
    return sequences

train_sequences = seq_generation(train_texts)
test_sequences = seq_generation(test_texts)
valid_sequences = seq_generation(valid_texts)

train_data = np.array(train_sequences).reshape((train_length, 60, 10))
valid_data = np.array(valid_sequences).reshape((valid_length, 60, 10))
test_data = np.array(test_sequences).reshape((test_length, 60, 10))

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
valid_labels = np.array(valid_labels)

embedding_matrix = np.load('embedding_matrix.npy')

#
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
#
# print('Preparing embedding matrix.')

# prepare embedding matrix
# num_words = min(MAX_NB_WORDS, len(word_index))
# embedding_matrix = np.zeros((MAX_NB_WORDS + 2, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i >= MAX_NB_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
# np.save('embedding_matrix.npy', embedding_matrix)

embedding_layer = Embedding(MAX_NB_WORDS + 2,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LINE_LENGTH,
                            trainable=True)


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.hidden_dim = attention_size
        super(AttLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer = 'he_normal', trainable=True)
        self.bw = self.add_weight(shape=(self.hidden_dim,), initializer = 'zero', trainable=True)
        self.uw = self.add_weight(shape=(self.hidden_dim,), initializer = 'he_normal', trainable=True)
        self.trainable_weights = [self.W, self.bw, self.uw]
        super(AttLayer,self).build(input_shape)

    def call(self, x, mask=None):
        x_reshaped = tf.reshape(x, [K.shape(x)[0]*K.shape(x)[1], K.shape(x)[-1]])
        ui = K.tanh(K.dot(x_reshaped, self.W) + self.bw)
        intermed = tf.reduce_sum(tf.multiply(self.uw, ui), axis=1)

        weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x)[0], K.shape(x)[1]]), dim=-1)
        weights = tf.expand_dims(weights, axis=-1)

        weighted_input = x*weights
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])


sentence_input = Input(shape=(MAX_LINE_LENGTH,), dtype='int32', name='sentence_input')
embedded_sequences = embedding_layer(sentence_input)
l_gru = Bidirectional(GRU(n_hidden, return_sequences=True, init='he_normal', inner_init='he_normal', inner_activation='sigmoid'), name='bidirect_word')(embedded_sequences)
l_att = AttLayer()(l_gru)
l_drop = Dropout(dropout)(l_att)
sentEncoder = Model(sentence_input, l_drop, name='sentence_encoder')

review_input = Input(shape=(MAX_SONG_LENGTH,MAX_LINE_LENGTH), dtype='int32', name='review_input')
review_encoder = TimeDistributed(sentEncoder, name='review_encoder')(review_input)
l_gru_sent = Bidirectional(GRU(n_hidden, return_sequences=True, init='he_normal', inner_init = 'he_normal', inner_activation = 'sigmoid'), name='bidirect_sentence')(review_encoder)
l_att_sent = AttLayer()(l_gru_sent)
l_drop_sent = Dropout(dropout)(l_att_sent)
preds = Dense(num_genres, activation='softmax')(l_drop_sent)
model = Model(review_input, preds)

optimizer = optimizers.RMSprop(lr=learning_rate, clipnorm = max_grad_norm)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# print "model fitting - Hierachical LSTM"
# print model.summary()
checkpointer = ModelCheckpoint(filepath='results/bidirectGRUattention.{epoch:02d}-{val_loss:.2f}.hdf5',verbose=1,save_best_only=True)
model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels),
          nb_epoch=training_epochs, batch_size=batch_size, callbacks=[checkpointer])
model.save('results/HAN-L.h5')
train_model=Model(review_input,l_drop_sent)
train=train_model.predict(train_data)
np.save('train_HAN.npy',train)

valid_model=Model(review_input,l_drop_sent)
valid=valid_model.predict(valid_data)
np.save('valid_HAN.npy',valid)

test_model=Model(review_input,l_drop_sent)
test=test_model.predict(test_data)
np.save('test_HAN.npy',test)

print('finish HAN')