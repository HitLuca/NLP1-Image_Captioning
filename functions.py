import json
import numpy as np
from keras.layers import Dense, RepeatVector
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


def create_image_model(max_caption_len):
    # This creates the image_model
    input_dim = 4096
    encoded_dim = 300

    encoder = Sequential()
    encoder.add(Dense(encoded_dim, activation='linear', input_shape=(input_dim,), trainable=False))
    encoder.load_weights('layers/autoencoder_weights')
    encoder.add(RepeatVector(max_caption_len-1))
    return encoder

