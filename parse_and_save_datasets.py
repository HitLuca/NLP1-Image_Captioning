import json

train_captions_filepath = 'dataset/merged_train.json'
val_captions_filepath = 'dataset/merged_val.json'

with open(train_captions_filepath) as f:
    train_captions_and_images = json.load(f)
with open(val_captions_filepath) as f:
    val_captions_and_images = json.load(f)

train_captions = []
val_captions = []

for i in range(len(train_captions_and_images)):
    print(i, '/', len(train_captions_and_images))
    captions = train_captions_and_images[i][1]
    for j in range(len(captions)):
        if captions[j][-1] != '.':
            captions[j].append('.')
        captions[j] = [x.lower() for x in captions[j]]
        captions[j].insert(0, '<S>')
        captions[j].append('</S>')
    train_captions.append(captions)

for i in range(len(val_captions_and_images)):
    print(i, '/', len(val_captions_and_images))
    captions = val_captions_and_images[i][1]
    for j in range(len(captions)):
        if captions[j][-1] != '.':
            captions[j].append('.')
        captions[j] = [x.lower() for x in captions[j]]
        captions[j].insert(0, '<S>')
        captions[j].append('</S>')
        val_captions.append(captions)

with open('dataset/train_captions.json', 'w') as f:
    json.dump(train_captions, f)

with open('dataset/val_captions.json', 'w') as f:
    json.dump(val_captions, f)
