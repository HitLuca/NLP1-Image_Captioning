import json
import numpy as np

captions_filepath = 'dataset/merged_val.json'
features_filepath = 'dataset/merged_val.npy'

with open(captions_filepath) as f:
    captions_and_filepaths = json.load(f)
features = np.load(features_filepath)

feature_master = []
caption_master = []

for i in range(len(captions_and_filepaths)):
    if i%1000 == 0:
        print(i, '/', len(captions_and_filepaths))
    captions = captions_and_filepaths[i][1]
    for caption in captions:
        caption = [x.lower() for x in caption]
        caption.insert(0, '<S>')
        caption.append('</S>')
        caption_master.append(caption)
        feature_master.append(features[i][:])

with open('dataset/val_captions.json', 'w') as f_captions:
    json.dump(caption_master, f_captions)

spanned_feature = np.array(feature_master)
np.save('dataset/val_features.npy', spanned_feature)
