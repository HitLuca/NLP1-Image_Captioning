import json
import numpy as np
from keras.engine import Merge
from keras.layers import Dense, RepeatVector, Embedding, LSTM, TimeDistributed
from keras.models import Sequential

from vgg_net import VGG16


def get_vgg_16():
    # We import the VGG-16 CNN
    vgg_16 = VGG16(include_top=True, weights='imagenet')

    # We remove the last layer, as we need 4096 features instead of 1000
    vgg_16.layers.pop()
    vgg_16.outputs = [vgg_16.layers[-1].output]
    return vgg_16


def get_captions(captions_filepath):
    return np.load(captions_filepath)


def get_features(features_filepath):
    return np.load(features_filepath)


def get_word_indexes(word_indexes_filepath):
    with open(word_indexes_filepath) as f:
        return json.load(f)


def create_image_model(max_caption_len, weights):
    # This creates the image_model
    input_dim = 4096
    encoded_dim = 300

    encoder = Sequential()
    encoder.add(Dense(encoded_dim, activation='linear', input_shape=(input_dim,), trainable=False))
    encoder.load_weights(weights)
    encoder.add(RepeatVector(max_caption_len-1))
    return encoder


def create_model(max_caption_len, word_indexes, embedding_dim):
    embedding_matrix_filepath = 'layers/embedding_matrix.npy'
    encoder_weights_filepath = 'layers/autoencoder_weights'

    image_model = create_image_model(max_caption_len, encoder_weights_filepath)
    embedding_matrix = np.load(embedding_matrix_filepath)

    embedding_layer = Embedding(len(word_indexes), embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_caption_len - 1,
                                trainable=False)

    language_model = Sequential()
    language_model.add(embedding_layer)

    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dense(len(word_indexes), activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    return model
