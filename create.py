import os
import json

images_dir = './test/images'
labels_dir = './test/labels'

ground_truth = {}

for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        label_file = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    ground_truth[filename] = 1
                else:
                    ground_truth[filename] = 0
        else:
            ground_truth[filename] = 0

with open('ground_truth_train.json', 'w') as f:
    json.dump(ground_truth, f, indent=2)

print("✅ ground_truth_train.json created successfully.")
