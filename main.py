import json

import numpy as np
from keras.engine import Merge
from keras.layers import Dense, LSTM, Activation, Embedding, K, TimeDistributed, RepeatVector
from keras.models import Sequential
import os

from vgg_net import VGG16


def get_vgg_16():
    # We import the VGG-16 CNN
    vgg_16 = VGG16(include_top=True, weights='imagenet')

    # We remove the last layer, as we need 4096 features instead of 1000
    vgg_16.layers.pop()
    vgg_16.outputs = [vgg_16.layers[-1].output]
    return vgg_16


def get_data(train_captions_filepath, train_features_filepath, val_captions_filepath, val_features_filepath):
    with open(train_captions_filepath) as f:
        train_captions = json.load(f)
    train_features = np.load(train_features_filepath)

    with open(val_captions_filepath) as f:
        val_captions = json.load(f)
    val_features = np.load(val_features_filepath)

    return train_captions, train_features, val_captions, val_features


def get_val_data(val_captions_filepath, val_features_filepath):

    with open(val_captions_filepath) as f:
        val_captions = json.load(f)
    val_features = np.load(val_features_filepath)

    return val_captions, val_features


# use this function so that the captions doesn't load again.
def get_features(train_features_filepath, val_features_filepath):

    train_features = np.load(train_features_filepath)
    val_features = np.load(val_features_filepath)

    return train_features, val_features


def span_drop_captions(captions, max_caption_len, filename_cap):

    for i in range(len(captions)):
        # process through each line of caption, 1 line = 1 caption
        if len(captions[i]) < max_caption_len:              # add extra span token, fill to max length
            for _ in range(max_caption_len - len(captions[i])):
                captions[i].append('</SPAN>')

    with open(filename_cap, 'w') as f:
        json.dump(captions, f)



def convert_captions(captions, word_indexes, filename):

    for i in range(len(captions)):
        captions[i] = [word_indexes[x] for x in captions[i]]

    with open(filename, 'w') as f:
        json.dump(captions, f)


def get_sequences_from_caption(caption, max_caption_len, word_indexes):
    i = 0
    sequences = []
    predictions = []
    while caption[i] != word_indexes['</S>']:
        l = []
        j = 0
        while j < i+1:
            l.append(caption[j])
            j += 1
        while j < max_caption_len:
            l.append(word_indexes['</SPAN>'])
            j += 1

        sequences.append(l)
        predictions.append([caption[i+1]])
        i += 1
    return sequences, predictions


def create_dataset(captions):
    C = []
    Y = []

    for i in range(len(captions)):
        C.append(captions[i][:-1])
        Y.append(captions[i][1:])

    C = np.array(C)
    Y = np.array(Y)

    return C, Y


def calculate_word_indexes(train_captions, val_captions):
    indexes = {}
    curr_index = 0

    for i in range(len(train_captions)):
        captions = train_captions[i]
        for caption in captions:
            for word in caption:
                if word not in indexes:
                    indexes[word] = curr_index
                    curr_index += 1
    for i in range(len(val_captions)):
        captions = val_captions[i]
        for caption in captions:
            for word in caption:
                if word not in indexes:
                    indexes[word] = curr_index
                    curr_index += 1
    indexes['</SPAN>'] = curr_index
    return indexes



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

    embedding_matrix = np.zeros((len(word_indexes), embedding_dim))
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


def parse_and_span_captions(captions_and_filepaths, max_caption_len):
    captions = []
    for i in range(len(captions_and_filepaths)):
        caption = captions_and_filepaths[1]
        for j in range(len(caption)):
            captions.append(caption[j])
    return captions


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


train_captions_filepath = 'dataset/train_captions.json'
train_features_filepath = 'dataset/merged_train.npy'

val_captions_filepath = 'dataset/val_captions.json'
val_features_filepath = 'dataset/val_spanned_features.npy'

max_caption_len = 16  # 16 words plus start of sentence and end of sentence
vocab_size = 10000
embedding_dim = 300

print('Loading validation datasets...')
# train_captions, train_features, val_captions, val_features = get_data(train_captions_filepath,
#                                                                       train_features_filepath,
#                                                                       val_captions_filepath,
#                                                                       val_features_filepath)
# val_captions, val_features = get_val_data(val_captions_filepath, val_features_filepath)

# use this if captions is already processed
val_features = np.load(val_features_filepath)
print('Done\n')


print('Loading word indexes...')
with open('dataset/word_indexes.json') as f:
    word_indexes = json.load(f)
# print('Calculating word indexes...')
# word_indexes = calculate_word_indexes(train_captions, val_captions)
# with open('dataset/word_indexes.json', 'w') as f:
#      json.dump(word_indexes, f)
print('Done\n')


print('Converting captions to int...')
filename_train_converted_caption = 'dataset/train_converted_caption.json'
filename_val_converted_caption = 'dataset/val_converted_caption.json'
# convert_captions(train_captions, word_indexes, filename_train_converted_caption)

# convert_captions(val_captions, word_indexes, filename_val_converted_caption)

# with open(filename_train_converted_caption) as f:
#     train_captions = json.load(f)

# with open(filename_val_converted_caption) as f:     # load the captions that is already converted to INT
#     val_captions = json.load(f)
print('Done\n')


# print('Dropping and spanning captions to length ' + str(max_caption_len) + '...')
filename_train_drop_span_cap = 'dataset/train_drop_span_cap.json'
filename_val_drop_span_cap = 'dataset/val_drop_span_captions.json'
# ***need to drop the features also***
# train_captions = span_drop_captions(train_captions, max_caption_len, filename_train_drop_span_cap)
#
# span_drop_captions(val_captions, max_caption_len, filename_val_drop_span_cap)

# with open(filename_train_drop_span_cap) as f:
#     print('Loading spanned, dropped training captions...')
#     train_captions = json.load(f)

with open(filename_val_drop_span_cap) as f:
    print('Loading spanned, dropped validation captions...')
    val_captions = json.load(f)
print('Done\n')

print('Creating training datasets...')
C, Y = create_dataset(val_captions)
train_captions = None
train_features = None
val_captions = None
print('Done\n')
print len(val_features)
#
# # This is the VGG-16 CNN, just because we can :P
# # vgg_16_net = get_vgg_16()
#
# # This creates the 4096->300 encoder
print('Loading encoder...')
image_model = create_image_model(max_caption_len-1)
print('Done\n')
#
# # This creates the word_indexes->300 embedding model
print('Creating embedding layer...')
# embedding_matrix = create_embedding_matrix(word_indexes, embedding_dim)
# np.savetxt('embedding_matrix', embedding_matrix)
embedding_matrix = np.loadtxt('embedding_matrix')
embedding_layer = Embedding(len(word_indexes), embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_caption_len-1,
                            trainable=False)
print('Done\n')
#
# # 512 hidden units in LSTM layer. 300-dimensional word vectors.
language_model = Sequential()
language_model.add(embedding_layer)

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(300, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#
# # prediction = image_model.predict([F])
# # print('Image model output shape:', prediction.shape)
# # prediction2 = language_model.predict([np.reshape(C, (C.shape[0], 16))])
# # print('Language model output shape:', prediction2.shape)
# # prediction3 = model.predict([F, np.reshape(C, (C.shape[0], 16))])
# # print('Main model output shape:', prediction3.shape)

print val_features.shape
print C.shape[0]
print Y.shape

# how to embed Y into tensor?????
model.fit([val_features, C], np.reshape(Y, (Y.shape[0], Y.shape[1], 300)), nb_epoch=5)

# json_string = model.to_json()
# with open('model_v1.json', 'w') as outfile:
#     json.dump(json_string, outfile)
#
# # Saving the weights
# model.save_weights('model_v1.h5')

