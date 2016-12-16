import json
import numpy as np
from keras.engine import Merge
from keras.layers import Dense, RepeatVector, Embedding, LSTM, TimeDistributed, Masking
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


def create_model(max_caption_len, word_indexes, embedding_dim, feature_dim, encoded_dim):
    language_model = Sequential()
    language_model.add(Embedding(len(word_indexes), embedding_dim, input_length=max_caption_len - 1, mask_zero=True))

    image_model = Sequential()
    image_model.add(Dense(encoded_dim, activation='linear', input_shape=(feature_dim,), trainable=True))
    image_model.add(RepeatVector(max_caption_len - 1))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(len(word_indexes), activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    return model
