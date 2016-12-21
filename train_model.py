import numpy as np
from keras.callbacks import ModelCheckpoint

from functions import get_captions, get_features, get_word_indexes, create_model_3

train_captions_filepath = 'dataset/frequencies/train_captions_4.npy'
train_features_filepath = 'dataset/frequencies/train_features_4.npy'

val_captions_filepath = 'dataset/frequencies/val_captions.npy'
val_features_filepath = 'dataset/frequencies/val_features.npy'

word_indexes_filepath = 'dataset/frequencies/word_indexes.json'
embedding_matrix_filepath = 'layers/embedding_matrix.npy'

max_caption_len = 19
feature_dim = 4096
embedding_dim = 300

word_indexes = get_word_indexes(word_indexes_filepath)

model = create_model_3(max_caption_len, word_indexes, embedding_dim, feature_dim, embedding_matrix_filepath)

checkpointer = ModelCheckpoint(filepath="model_weights.hdf5", verbose=0)

model.load_weights('model_weights.hdf5')


for i in range(5, 8):
    captions = 'dataset/frequencies/train_captions_' + str(i) + '.npy'
    features = 'dataset/frequencies/train_features_' + str(i) + '.npy'

    print(captions)
    print(features)

    train_captions = get_captions(captions)
    train_features = get_features(features)

    X_captions = train_captions[:, :-1]
    Y = np.expand_dims(train_captions[:, 1:], -1)
    X = [train_features, X_captions]

    try:
        model.fit(X, Y, nb_epoch=1, callbacks=[checkpointer], shuffle=True)
    except KeyboardInterrupt:
        checkpointer.on_epoch_end(0)
