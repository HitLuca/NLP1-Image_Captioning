import json

import numpy as np
from keras.engine import Input
from keras.engine import Merge
from keras.engine import Model
from keras.layers import Dense, LSTM, Activation, Embedding, K
from keras.models import Sequential
from keras.optimizers import Adam
import os

from vgg_net import VGG16


def get_vgg_16():
    # We import the VGG-16 CNN
    vgg_16 = VGG16(include_top=True, weights='imagenet')

    # We remove the last layer, as we need 4096 features instead of 1000
    vgg_16.layers.pop()
    vgg_16.outputs = [vgg_16.layers[-1].output]
    return vgg_16


def create_embedding_matrix(word_indexes, embedding_dim):
    embeddings_index = {}
    f = open(os.path.join('embeddings/glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_indexes) + 1, embedding_dim))
    for word, i in word_indexes.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_image_model():
    # This creates the image_model
    image_model = create_autoencoder(4096, 300)
    image_model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return image_model


def create_autoencoder(input_dim, encoded_dim):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoded_dim, activation='linear')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    return Model(input=input_img, output=decoded)


def get_data():
    # 20000 images
    # 4096 features

    captions_filepath = 'dataset/merged_val.json'
    features_filepath = 'dataset/merged_val.npy'

    with open(captions_filepath) as f:
        captions = json.load(f)
    features = np.load(features_filepath)

    return captions, features


def calculate_word_indexes(captions):
    indexes = {}
    curr_index = 1

    for i in range(len(captions)):
        caption = captions[i]
        for word in caption:
            if word not in indexes.keys():
                indexes[word] = curr_index
                curr_index += 1
    return indexes


def span_captions_features(paths_and_captions, features):
    captions = []
    spanned_features = []
    for i in range(len(paths_and_captions)):
        caption = paths_and_captions[i]
        for j in range(len(caption[1])):
            sentence = caption[1][j]
            captions.append(np.array(sentence))
            spanned_features.append(features[i])
    return captions, spanned_features


def convert_and_drop_captions(captions, features, max_length, word_indexes):
    new_captions = []

    i = 0
    while i < len(captions):
        caption = captions[i]

        # Pop caption if longer than max_length
        if len(caption) > max_length:
            captions.pop(i)
            features.pop(i)
        else:
            sentence = []
            j = 0
            while j < len(caption):
                sentence.append(word_indexes[caption[j]])
                j += 1

            while j < max_length:
                sentence.append(0)
                j += 1

            new_captions.append(sentence)
            i += 1
    return new_captions


longest_caption = 49  # The length of the longest caption
max_caption_len = 16
vocab_size = 10000
embedding_dim = 300

# We load the data and create the training set
print('Loading and spanning data...')
paths_and_captions, features = get_data()
captions, spanned_features = span_captions_features(paths_and_captions[:100], features[:100])
print('Done')

# We don't need this anymore
paths_and_captions = None
features = None

word_indexes = calculate_word_indexes(captions)

print('Converting and dropping longer captions...')
captions = convert_and_drop_captions(captions, spanned_features, max_caption_len, word_indexes)
print('Done')

# This is the VGG-16 CNN, just because we can :P
# vgg_16_net = get_vgg_16()

# This creates the 4096->300 encoder
# image_model = create_image_model()

# This creates the 10k->300 embedding model
print('Creating embedding layer...')
# embedding_matrix = create_embedding_matrix(word_indexes, embedding_dim)
# np.savetxt('embedding_matrix', embedding_matrix)
embedding_matrix = np.loadtxt('embedding_matrix')
embedding_layer = Embedding(len(word_indexes) + 1, embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_caption_len,
                            trainable=False)
# embedding_model = Sequential()
# embedding_model.add(embedding_layer)
# print(embedding_model.predict(np.array([captions[0]]))[0].shape)
print('Done')

# 512 hidden units in LSTM layer. 300-dimensional word vectors.
language_model = Sequential()
language_model.add(embedding_layer)
language_model.add(LSTM(512, return_sequences=False, input_shape=(max_caption_len, embedding_dim)))
language_model.add(Dense(len(word_indexes) + 1))
# language_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# language_model.fit(np.array(captions[:-1]), np.array(captions[1:]), batch_size=16, nb_epoch=10)

#
# model = Sequential()
# model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
# model.add(Dense(vocab_size))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#
# model.fit(np.array(spanned_features), np.array(captions), batch_size=16, nb_epoch=10)








# language_model = Sequential()
# language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
# language_model.add(GRU(output_dim=128, return_sequences=True))
# language_model.add(Dense(vocab_size))
# language_model.add(Activation('softmax'))
#
# language_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# language_model.fit(features, captions, batch_size=16, nb_epoch=100)
