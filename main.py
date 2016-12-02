import json

import numpy as np
from keras.engine import Input
from keras.engine import Merge
from keras.engine import Model
from keras.layers import Dense, LSTM, Activation, Embedding, K, TimeDistributed, RepeatVector
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


def create_image_model(max_caption_len):
    # This creates the image_model
    input_dim = 4096
    encoded_dim = 300

    encoder = Sequential()
    encoder.add(Dense(encoded_dim, activation='linear', input_shape=(input_dim,), trainable=False))
    encoder.load_weights('autoencoder_weights')
    encoder.add(RepeatVector(max_caption_len))
    return encoder


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
    curr_index = 3

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
            sentence.append(0)
            j = 1
            while j < len(caption):
                sentence.append(word_indexes[caption[j-1]])
                j += 1

            sentence.append(1)

            while j < max_length:
                sentence.append(2)
                j += 1

            new_captions.append(sentence)
            i += 1
    return new_captions


longest_caption = 49  # The length of the longest caption
max_caption_len = 16  # 16 words plus start of sentence and end of sentence
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
print('Loading encoder...')
image_model = create_image_model(max_caption_len)
print('Done')

# This creates the 10k->300 embedding model
print('Creating embedding layer...')
# embedding_matrix = create_embedding_matrix(word_indexes, embedding_dim)
# np.savetxt('embedding_matrix', embedding_matrix)
embedding_matrix = np.loadtxt('embedding_matrix')
embedding_layer = Embedding(len(word_indexes)+1, embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_caption_len,
                            trainable=False)
print('Done')

# 512 hidden units in LSTM layer. 300-dimensional word vectors.
language_model = Sequential()
language_model.add(embedding_layer)
language_model.add(LSTM(512, return_sequences=True, input_shape=(max_caption_len, embedding_dim)))
language_model.add(TimeDistributed(Dense(300)))

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
model.add(LSTM(256, return_sequences=False))
model.add(Dense(len(word_indexes)+1))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

caption = [0,2,3,3,3,3,3,3,3,3,3,3,3,3,3,1]

# prediction = image_model.predict(np.array([spanned_features[0]]))
# print('Image model output shape:', prediction.shape)
# prediction2 = language_model.predict(np.array([caption]))
# print('Language model output shape:', prediction2.shape)
# prediction3 = model.predict([np.array([spanned_features[0]]), np.array([caption])])
# print('Main model output shape:', prediction3.shape)

output = np.append(np.zeros(len(word_indexes)), [1], axis=0)
model.fit([np.array([spanned_features[0]]), np.array([caption])], [np.reshape(output, (1, 974))], nb_epoch=10000)
# model.fit(np.array(spanned_features), np.array(captions), batch_size=16, nb_epoch=10)








# language_model = Sequential()
# language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
# language_model.add(GRU(output_dim=128, return_sequences=True))
# language_model.add(Dense(vocab_size))
# language_model.add(Activation('softmax'))
#
# language_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# language_model.fit(features, captions, batch_size=16, nb_epoch=100)
