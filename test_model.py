from keras.preprocessing import image
import numpy as np
import keras.backend as K

from functions import get_vgg_16, create_model, get_word_indexes


def get_features(vgg16, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return vgg16.predict(x)


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


word_indexes_filepath = 'dataset/word_indexes.json'

max_caption_len = 19
feature_dim = 4096
encoded_dim = 300
embedding_dim = 300

word_indexes = get_word_indexes(word_indexes_filepath)
model = create_model(max_caption_len, word_indexes, embedding_dim, feature_dim, encoded_dim)
model.load_weights('model_weights.hdf5')


vgg16 = get_vgg_16()

img_path = 'img/table.jpg'
image_features = get_features(vgg16, img_path)

caption_index = 1
caption = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
finished = False

while not finished:
    prediction = model.predict([image_features, np.array(caption).reshape((1, 18))])
    prediction = prediction[0]
    index = np.argmax(prediction[caption_index-1])
    word = list(word_indexes.keys())[list(word_indexes.values()).index(index)]
    caption[caption_index] = index
    caption_index += 1
    if word == '</S>' or caption_index == max_caption_len-1:
        finished = True

i = 1
while i < max_caption_len-1 and caption[i] != word_indexes['</S>']:
    print(list(word_indexes.keys())[list(word_indexes.values()).index(caption[i])], end=' ')
    i += 1
print()
