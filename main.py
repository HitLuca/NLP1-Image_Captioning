import numpy as np
from keras.engine import Merge
from keras.layers import Dense, LSTM, Embedding, TimeDistributed
from keras.models import Sequential

from functions import create_image_model, get_captions, get_features, get_word_indexes

train_captions_filepath = 'dataset/processed/train_captions.npy'
train_features_filepath = 'dataset/processed/train_features.npy'

val_captions_filepath = 'dataset/processed/val_captions.npy'
val_features_filepath = 'dataset/processed/val_features.npy'

word_indexes_filepath = 'dataset/word_indexes.json'
embedding_matrix_filepath = 'layers/embedding_matrix.npy'

max_caption_len = 19
vocab_size = 10000
embedding_dim = 300

print('Loading datasets...')
# train_captions = get_captions(train_captions_filepath)
# train_features = get_features(train_features_filepath)

val_captions = get_captions(val_captions_filepath)
val_features = get_features(val_features_filepath)

print('Loading word indexes...')
word_indexes = get_word_indexes(word_indexes_filepath)

print('Creating training datasets...')
X_captions = val_captions[:, :-1]
Y = val_captions[:, 1:]
X = [val_features, X_captions]
val_captions = None

# This is the VGG-16 CNN, just because we can :P
# vgg_16_net = get_vgg_16()

# This creates the 4096->300 encoder
print('Loading encoder...')
image_model = create_image_model(max_caption_len)

print('Creating embedding layer...')
embedding_matrix = np.load(embedding_matrix_filepath)

embedding_layer = Embedding(len(word_indexes), embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_caption_len-1,
                            trainable=False)
embedding_matrix = None
print('Done\n')

language_model = Sequential()
language_model.add(embedding_layer)

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(len(word_indexes), activation='softmax')))
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')


model.fit(X, np.expand_dims(Y, -1), nb_epoch=50)
