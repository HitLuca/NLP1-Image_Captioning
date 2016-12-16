import numpy as np
from keras.callbacks import ModelCheckpoint

from functions import get_captions, get_features, get_word_indexes, create_model

train_captions_filepath = 'dataset/processed/train_captions.npy'
train_features_filepath = 'dataset/processed/train_features.npy'

val_captions_filepath = 'dataset/processed/val_captions.npy'
val_features_filepath = 'dataset/processed/val_features.npy'

word_indexes_filepath = 'dataset/word_indexes.json'

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
Y = np.expand_dims(val_captions[:, 1:], -1)
X = [val_features, X_captions]

# This is the VGG-16 CNN, just because we can :P
# vgg_16_net = get_vgg_16()

print('Creating model...')
model = create_model(max_caption_len, word_indexes, embedding_dim)

checkpointer = ModelCheckpoint(filepath="model_weights.hdf5", verbose=0)

# model.load_weights('model_weights.hdf5')

model.fit(X, np.expand_dims(Y, -1), nb_epoch=50, callbacks=[checkpointer])

# prediction = model.predict([val_features[0].reshape((1, 4096)), np.array([1, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58]).reshape((1, 18))])
# prediction = prediction[0]
# print(prediction.shape)
