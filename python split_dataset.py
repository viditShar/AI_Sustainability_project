import os
import shutil
import random

# Paths
base_dir = r"C:\Users\asus\OneDrive\Desktop\Projects\AI\Sustainability\dataset-resized\dataset-resized"
target_dir = r"C:\Users\asus\OneDrive\Desktop\Projects\AI\Sustainability\dataset"

# Split ratio
train_ratio = 0.8  # 80% train, 20% test

# Create folders
for split in ['train', 'test']:
    split_path = os.path.join(target_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Go through each category
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if not os.path.isdir(category_path):
        continue

    # Create category folders
    os.makedirs(os.path.join(target_dir, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test', category), exist_ok=True)

    # List and shuffle images
    images = os.listdir(category_path)
    random.shuffle(images)

    # Split
    train_count = int(len(images) * train_ratio)
    train_files = images[:train_count]
    test_files = images[train_count:]

    # Move files
    for img in train_files:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(target_dir, 'train', category, img))
    for img in test_files:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(target_dir, 'test', category, img))

print("✅ Dataset organized successfully!")
