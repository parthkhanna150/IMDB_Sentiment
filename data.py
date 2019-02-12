# copy contents of all files in both folders into a list
import glob
import os

train_data = []
test_data = []

# train data
train_neg = glob.glob(os.path.join(os.getcwd(), "Dataset/train/neg", "*.txt"))
for f_path in train_neg:
    with open(f_path) as f:
        train_data.append(f.read())

train_pos = glob.glob(os.path.join(os.getcwd(), "Dataset/train/pos", "*.txt"))
for f_path in train_pos:
    with open(f_path) as f:
        train_data.append(f.read())


# test data
test_files = glob.glob(os.path.join(os.getcwd(), "Dataset/test", "*.txt"))
for f_path in test_files:
    with open(f_path) as f:
        test_data.append(f.read())
