import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Masking, LSTM
from keras.models import Sequential

from functions import get_captions, get_features, get_word_indexes, create_model

train_captions_filepath = 'dataset/processed/train_captions.npy'
train_features_filepath = 'dataset/processed/train_features.npy'

val_captions_filepath = 'dataset/processed/val_captions.npy'
val_features_filepath = 'dataset/processed/val_features.npy'

word_indexes_filepath = 'dataset/word_indexes.json'

max_caption_len = 19
feature_dim = 4096
encoded_dim = 300
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
Y = np.expand_dims(val_captions[:, 1:], -1)
X = [val_features, X_captions]

# This is the VGG-16 CNN, just because we can :P
# vgg_16_net = get_vgg_16()

print('Creating model...')
model = create_model(max_caption_len, word_indexes, embedding_dim, feature_dim, encoded_dim)

checkpointer = ModelCheckpoint(filepath="model_weights.hdf5", verbose=0)

model.load_weights('model_weights.hdf5')

try:
    model.fit(X, Y, nb_epoch=1000, callbacks=[checkpointer], validation_split=0.2)
except KeyboardInterrupt:
    checkpointer.on_epoch_end(0)
