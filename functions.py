import json
import numpy as np
from keras.engine import Merge
from keras.layers import Dense, RepeatVector, Embedding, LSTM, TimeDistributed
from keras.models import Sequential

from vgg_net import VGG16


def get_vgg_16():
    vgg_16 = VGG16(include_top=True, weights='imagenet')

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


def create_model_1(max_caption_len, word_indexes, embedding_dim, feature_dim, encoded_dim):
    language_model = Sequential()
    language_model.add(Embedding(len(word_indexes), embedding_dim, input_length=max_caption_len - 1, mask_zero=True))
    language_model.add(LSTM(256, return_sequences=True))

    image_model = Sequential()
    # image_model.add(Dense(encoded_dim, activation='linear', input_shape=(feature_dim,)))
    image_model.add(RepeatVector(max_caption_len - 1, input_shape=(feature_dim,)))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(len(word_indexes), activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    return model


def create_model_2(max_caption_len, word_indexes, embedding_dim, feature_dim, encoded_dim):
    language_model = Sequential()
    language_model.add(Embedding(len(word_indexes), embedding_dim, input_length=max_caption_len - 1, mask_zero=True))

    image_model = Sequential()
    image_model.add(Dense(encoded_dim, activation='linear', input_shape=(feature_dim,)))
    image_model.add(RepeatVector(max_caption_len - 1, input_shape=(feature_dim,)))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(LSTM(1024, return_sequences=True))
    model.add(TimeDistributed(Dense(len(word_indexes), activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    return model


def create_model_3(max_caption_len, word_indexes, embedding_dim, feature_dim, embedding_matrix_filepath):
    embedding_matrix = np.load(embedding_matrix_filepath)

    language_model = Sequential()
    language_model.add(Embedding(len(word_indexes), embedding_dim, weights=[embedding_matrix], input_length=max_caption_len - 1, mask_zero=True))

    image_model = Sequential()
    # image_model.add(Dense(encoded_dim, activation='linear', input_shape=(feature_dim,)))
    image_model.add(RepeatVector(max_caption_len - 1, input_shape=(feature_dim,)))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dense(len(word_indexes), activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model