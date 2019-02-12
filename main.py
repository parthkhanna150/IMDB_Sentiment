# copy contents of all files in both folders into a list
import glob
import os
from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

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
# print(train_data[0])
# print(preprocess(train_data[0]))

# test data
test_files = glob.glob(os.path.join(os.getcwd(), "Dataset/test", "*.txt"))
for f_path in test_files:
    with open(f_path) as f:
        test_data.append(f.read())

# targets: first 12500 are pos, next 12500 are neg
targets = [1 if i<12500 else 0 for i in range(25000)]

train_data_clean = preprocess(train_data, advanced=False)
test_data_clean = preprocess(test_data, advanced=False)

# splitting the data
X_train, X_validation, y_train, y_validation = train_test_split(train_data, targets, train_size=0.8, test_size=0.2)
# print(train_data_clean[0])
# print(test_data_clean[0])
# print(y_train[0])

# Bag of Words vectorization
cv = CountVectorizer().fit(X_train)
X_train_counts = cv.transform(X_train)
X_validation_counts = cv.transform(X_validation)

# tfidf
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

# normalization
normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_validation_normalized = normalizer_tranformer.transform(X_validation_tfidf)

clf = MultinomialNB().fit(X_train_normalized, y_train)
y_pred = clf.predict(X_validation_normalized)

print(metrics.classification_report(y_validation, y_pred))
