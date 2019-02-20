# copy contents of all files in both folders into a list
import glob
import os
import csv
import numpy as np
from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV

# copy contents of all files in both folders into a list
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

def sort_nicely(l):
# Sort the given list in the way that humans expect.
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort(key=alphanum_key)

test_files = glob.glob(os.path.join(os.getcwd(), "Dataset/test", "*.txt"))
sort_nicely(test_files)
test_files_ids = [int(re.sub("[^0-9]","", item)) for item in test_files]

for f_path in test_files:
    with open(f_path) as f:
        test_data.append(f.read())

# targets: first 12500 are pos, next 12500 are neg
targets = [0 if i<12500 else 1 for i in range(25000)]

train_data_clean = preprocess(train_data)
test_data_clean = preprocess(test_data)

# save to a file if using above complex

X_train, X_validation, y_train, y_validation = train_test_split(train_data_clean, targets, train_size=0.8, test_size=0.2, random_state=1)

def display_results(y_val, y_pred):
    print(metrics.classification_report(y_val, y_pred))
    print("Accuracy % = ", metrics.accuracy_score(y_val, y_pred))
