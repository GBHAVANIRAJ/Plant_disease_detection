import os
import shutil
import random

dataset = "Dataset"
train_path = "Dataset/train"
valid_path = "Dataset/valid"

os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)

train_ratio = 0.8

for cls in os.listdir(dataset):
    cls_path = os.path.join(dataset, cls)

    if cls in ["train", "valid"]:
        continue

    if not os.path.isdir(cls_path):
        continue

    print(f"Splitting: {cls}")

    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(valid_path, cls), exist_ok=True)

    images = os.listdir(cls_path)
    random.shuffle(images)

    split = int(len(images) * train_ratio)

    train_imgs = images[:split]
    valid_imgs = images[split:]

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_path, cls, img))

    for img in valid_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(valid_path, cls, img))

print("✅ Dataset split completed!")
