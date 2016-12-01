import json

from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense
import numpy as np

def create_autoencoder(input_dim, encoded_dim):
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoded_dim, activation='linear')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder


def train_autoencoder(autoencoder):
    valid_filepath = 'dataset/merged_val.npy'
    train_filepath = 'dataset/merged_train.npy'

    print('Loading features...')
    train = np.load(train_filepath)
    valid = np.load(valid_filepath)
    print('Done')

    features = np.append(train, valid, axis=0)

    print('Training autoencoder...')
    autoencoder.fit(features,
                    features,
                    validation_split=0.2,
                    nb_epoch=3,
                    batch_size=25,
                    shuffle=True)
    print('Done')

autoencoder = create_autoencoder(4096, 300)
# train_autoencoder(autoencoder)
# autoencoder.save_weights('autoencoder_weights')
autoencoder.load_weights('complete_autoencoder_weights')
autoencoder.layers.pop()
autoencoder.outputs = [autoencoder.layers[-1].output]
autoencoder.save_weights('autoencoder_weights')