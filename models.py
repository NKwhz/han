from keras.layers import Input, Bidirectional, GRU, Dropout, TimeDistributed, Dense, Layer, BatchNormalization, Embedding
from keras.models import Model
from keras import optimizers
import tensorflow as tf
import keras.backend as K

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
MAX_SONG_LENGTH = 60

dropout = 0.5
attention_size=100
n_hidden=50
num_genres=9
learning_rate = .01 # tuned on dev set
training_epochs = 10
max_grad_norm = 1. # tuned on dev set
batch_size=64


def build_model(lyric_shape, sentence_shape, n_hidden, dropout, num_genres, learning_rate, max_grad_norm, attention_size, embedding_matrix):
    class AttLayer(Layer):
        def __init__(self, **kwargs):
            self.hidden_dim = attention_size
            super(AttLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='kernel1', shape=(input_shape[-1], self.hidden_dim), initializer='he_normal',
                                     trainable=True)
            self.bw = self.add_weight(name='kernel2', shape=(self.hidden_dim,), initializer='zero', trainable=True)
            self.uw = self.add_weight(name='kernel3', shape=(self.hidden_dim,), initializer='he_normal', trainable=True)
            self.trainable_weights = [self.W, self.bw, self.uw]
            super(AttLayer, self).build(input_shape)

        def call(self, x, mask=None):
            x_reshaped = tf.reshape(x, [K.shape(x)[0] * K.shape(x)[1], K.shape(x)[-1]])
            ui = K.tanh(K.dot(x_reshaped, self.W) + self.bw)
            intermed = tf.reduce_sum(tf.multiply(self.uw, ui), axis=1)

            weights = tf.nn.softmax(tf.reshape(intermed, [K.shape(x)[0], K.shape(x)[1]]), dim=-1)
            weights = tf.expand_dims(weights, axis=-1)

            weighted_input = x * weights
            return K.sum(weighted_input, axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[2])



    sentence_input = Input(shape=(MAX_LINE_LENGTH,), name='sentence_input')
    embedding_layer = Embedding(MAX_NB_WORDS + 2,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_LINE_LENGTH,
                                trainable=True)
    embedded_sequences = embedding_layer(sentence_input)
    l_gru = Bidirectional(
        GRU(n_hidden, return_sequences=True, kernel_initializer='he_normal', recurrent_initializer='he_normal', activation='sigmoid'),
        merge_mode='concat', name='bidirect_word')(embedded_sequences)
    l_att = AttLayer()(l_gru)
    l_drop = Dropout(dropout)(l_att)
    sentEncoder = Model(sentence_input, l_drop, name='sentence_encoder')

    review_input = Input(shape=lyric_shape, name='review_input')
    norm = BatchNormalization()(review_input)
    review_encoder = TimeDistributed(sentEncoder, name='review_encoder')(norm)
    l_gru_sent = Bidirectional(
        GRU(n_hidden, return_sequences=True, kernel_initializer='he_normal', recurrent_initializer='he_normal', activation='sigmoid'),
        merge_mode='concat', name='bidirect_sentence')(review_encoder)
    l_att_sent = AttLayer()(l_gru_sent)
    l_drop_sent = Dropout(dropout, name='rep')(l_att_sent)
    preds = Dense(num_genres, activation='sigmoid')(l_drop_sent)
    model = Model(review_input, preds)

    optimizer = optimizers.RMSprop(lr=learning_rate, clipnorm=max_grad_norm)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model