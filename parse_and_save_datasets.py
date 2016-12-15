import json
import numpy as np

train_captions_filepath = 'dataset/merged_train.json'
train_features_filepath = 'dataset/merged_train.npy'
val_captions_filepath = 'dataset/merged_val.json'
val_features_filepath = 'dataset/merged_val.npy'

max_caption_len = 16
#
# with open(train_captions_filepath) as f:
#     train_captions_and_images = json.load(f)
# train_features = np.load(train_features_filepath)

with open(val_captions_filepath) as f:
    val_captions_and_images = json.load(f)
val_features = np.load(val_features_filepath)

train_captions = []
val_captions = []

train_feature_master = []
val_feature_master = []

# for i in range(len(train_captions_and_images)):
#     print(i, '/', len(train_captions_and_images))
#     captions = train_captions_and_images[i][1]
#     for j in range(len(captions)):
#         if captions[j][-1] != '.':
#             captions[j].append('.')
#         captions[j] = [x.lower() for x in captions[j]]
#         captions[j].insert(0, '<S>')
#         captions[j].append('</S>')
#         train_captions.append(captions)
#         train_feature_master.append(train_features[i][:])

for i in range(len(val_captions_and_images)):
    if i%1000 == 0:
        print(i, '/', len(val_captions_and_images))
    captions = val_captions_and_images[i][1]
    for j in range(len(captions)):
        if captions[j][-1] != '.':
            captions[j].append('.')
        captions[j] = [x.lower() for x in captions[j]]
        captions[j].insert(0, '<S>')
        captions[j].append('</S>')
        if len(captions[j]) < max_caption_len:
            val_captions.append(captions[j])
            val_feature_master.append(val_features[i][:])

''' ===== training ===== '''
# with open('dataset/train_captions.json', 'w') as f:
#     json.dump(train_captions, f)

# training
# train_spanned_feature = np.array(train_feature_master)
# np.save('dataset/train_spanned_features.npy', train_spanned_feature)

print len(val_captions)
print len(val_feature_master)

''' ===== validation ===== '''
with open('dataset/val_captions.json', 'w') as f:
    json.dump(val_captions, f)

val_spanned_feature = np.array(val_feature_master)
np.save('dataset/val_spanned_features.npy', val_spanned_feature)
