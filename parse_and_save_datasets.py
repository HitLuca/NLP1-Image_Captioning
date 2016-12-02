import json

import numpy as np

captions_filepath = 'dataset/merged_val.json'
features_filepath = 'dataset/merged_val.npy'

with open(captions_filepath) as f:
    captions_and_filepaths = json.load(f)
features = np.load(features_filepath)

f_features = open('dataset/val_features', 'a')
f_captions = open('dataset/val_captions', 'w')

new_features = np.array([])
for i in range(len(captions_and_filepaths)):
    print(i, '/', len(captions_and_filepaths))
    captions = captions_and_filepaths[i][1]
    for caption in captions:
        if caption[-1] != '.':
            caption.append('.')
        caption = [x.lower() for x in caption]
        caption.insert(0, '<S>')
        caption.append('</S>')
        print(caption, file=f_captions)
        new_features = np.append(new_features, features[i])
    f_captions.flush()
f_captions.close()
new_features.tofile(f_features)
