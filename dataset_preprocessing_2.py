import json
import numpy as np

from functions import get_features


def get_captions(captions_filepath):
    with open(captions_filepath) as f:
        captions = json.load(f)
    return captions


def add_sentence_tokens(captions):
    for i in range(len(captions)):
        captions[i] = captions[i][1]
        for j in range(len(captions[i])):
            if captions[i][j][-1] != '.':
                captions[i][j].append('.')
            captions[i][j] = [x.lower() for x in captions[i][j]]
            captions[i][j].insert(0, '<S>')
            captions[i][j].append('</S>')


def calculate_word_frequencies(train_captions, val_captions):
    word_frequencies = {}

    for i in range(len(train_captions)):
        captions = train_captions[i]
        for caption in captions:
            for word in caption:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    for i in range(len(val_captions)):
        captions = val_captions[i]
        for caption in captions:
            for word in caption:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    total_1 = 0
    total_2 = 0
    total_3 = 0
    total_4 = 0

    for word in word_frequencies:
        if word_frequencies[word] <= 1:
            total_1 += 1
        if word_frequencies[word] <= 2:
            total_2 += 1
        if word_frequencies[word] <= 3:
            total_3 += 1
        if word_frequencies[word] <= 4:
            total_4 += 1

    print('less than 2', total_1)
    print('less than 3', total_2)
    print('less than 4', total_3)
    print('less than 5', total_4)
    print()
    print('total', len(word_frequencies))
    return word_frequencies


def remove_low_frequencies(captions, word_frequencies):
    for i in range(len(captions)):
        for j in range(len(captions[i])):
            for k in range(len(captions[i][j])):
                if word_frequencies[captions[i][j][k]] <= 3:
                    captions[i][j][k] = '</?>'


def calculate_word_indexes(train_captions, val_captions):
    indexes = {}

    indexes['</SPAN>'] = 0
    curr_index = 1

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

    return indexes


def span_drop_captions_features(captions, features, max_caption_len, word_indexes):
    new_captions = []
    new_features = []

    feature_index = 0

    for feature_captions in captions:
        for caption in feature_captions:
            if (len(caption)) <= max_caption_len:
                for _ in range(max_caption_len - len(caption)):
                    caption.append('</SPAN>')
                for i in range(len(caption)):
                    caption[i] = word_indexes[caption[i]]
                new_captions.append(caption)
                new_features.append(features[feature_index][:])
        feature_index += 1

    new_captions = np.array(new_captions)
    new_features = np.array(new_features)
    return new_captions, new_features


def create_embedding_matrix(word_indexes, embedding_dim, embeddings_filepath):
    embeddings_index = {}
    with open(embeddings_filepath) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_indexes), embedding_dim))
    for word, i in word_indexes.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            temp = np.empty(embedding_dim)
            temp.fill(word_indexes[word])
            embedding_matrix[i] = temp

    return embedding_matrix


train_captions_filepath = 'dataset/original/merged_train.json'
train_features_filepath = 'dataset/original/merged_train.npy'

val_captions_filepath = 'dataset/original/merged_val.json'
val_features_filepath = 'dataset/original/merged_val.npy'

processed_train_captions_filepath = 'dataset/frequencies/train_captions_7.npy'
processed_train_features_filepath = 'dataset/frequencies/train_features_7.npy'

processed_val_captions_filepath = 'dataset/frequencies/val_captions.npy'
processed_val_features_filepath = 'dataset/frequencies/val_features.npy'

embeddings_filepath = 'embeddings/glove.6B.300d.txt'
embedding_matrix_filepath = 'layers/embedding_matrix.npy'

embedding_dim = 300
max_caption_len = 19

train_captions = get_captions(train_captions_filepath)
train_features = get_features(train_features_filepath)

train_captions = train_captions[120000:]
train_features = train_features[120000:]

# val_captions = get_captions(val_captions_filepath)
# val_features = get_features(val_features_filepath)


add_sentence_tokens(train_captions)
# add_sentence_tokens(val_captions)

# word_frequencies = calculate_word_frequencies(train_captions, val_captions)
# with open('dataset/frequencies/word_frequencies.json', 'w') as f:
#     json.dump(word_frequencies, f)

with open('dataset/frequencies/word_frequencies.json', 'r') as f:
    word_frequencies = json.load(f)

remove_low_frequencies(train_captions, word_frequencies)
# remove_low_frequencies(val_captions, word_frequencies)

# word_indexes = calculate_word_indexes(train_captions, val_captions)
# with open('dataset/frequencies/word_indexes.json', 'w') as f:
#     json.dump(word_indexes, f)
# train_captions = None
#
with open('dataset/frequencies/word_indexes.json', 'r') as f:
    word_indexes = json.load(f)

# embedding_matrix = create_embedding_matrix(word_indexes, embedding_dim, embeddings_filepath)
# with open(embedding_matrix_filepath, 'wb') as f:
#     np.save(f, embedding_matrix)
# embedding_matrix = None

train_captions, train_features = span_drop_captions_features(train_captions,
                                                             train_features,
                                                             max_caption_len,
                                                             word_indexes)

train_captions = train_captions.astype(int)

with open(processed_train_captions_filepath, 'wb') as f:
    np.save(f, train_captions)

with open(processed_train_features_filepath, 'wb') as f:
    np.save(f, train_features)


# val_captions, val_features = span_drop_captions_features(val_captions, val_features,  max_caption_len, word_indexes)
#
# val_captions = val_captions.astype(int)
#
# with open(processed_val_captions_filepath, 'wb') as f:
#     np.save(f, val_captions)
#
# with open(processed_val_features_filepath, 'wb') as f:
#     np.save(f, val_features)


