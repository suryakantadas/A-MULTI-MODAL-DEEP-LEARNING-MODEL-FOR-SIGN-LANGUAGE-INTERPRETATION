# split_dataset.py

import os
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset_landmark_sequences"

X = []
y = []
label_map = {}  # Maps class name to integer label

# Step 1: Read each gesture class
for label_index, class_name in enumerate(sorted(os.listdir(DATASET_DIR))):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    label_map[class_name] = label_index  # Save label

    for seq_file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, seq_file)
        data = np.load(file_path)
        X.append(data)
        y.append(label_index)

# Convert to Numpy Arrays
X = np.array(X)
y = np.array(y)

print(f"Total samples loaded: {len(X)}")

# Step 2: Split 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Step 3: Save splits
os.makedirs("splits/train", exist_ok=True)
os.makedirs("splits/val", exist_ok=True)
os.makedirs("splits/test", exist_ok=True)

np.save("splits/train/X_train.npy", X_train)
np.save("splits/train/y_train.npy", y_train)
np.save("splits/val/X_val.npy", X_val)
np.save("splits/val/y_val.npy", y_val)
np.save("splits/test/X_test.npy", X_test)
np.save("splits/test/y_test.npy", y_test)

# Optional: Save label mapping
with open("splits/labels.txt", "w") as f:
    for name, idx in label_map.items():
        f.write(f"{idx}: {name}\n")

print("Dataset split complete and saved.")
