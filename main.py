# copy contents of all files in both folders into a list
import glob
import os
import csv
from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC

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

# print(train_data[0])
# print(preprocess(train_data[0]))

# test data
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

with open('english') as f:
    stopwords = f.read().splitlines()

with open('prep_train.txt') as f:
        train_data_clean = f.read().splitlines()

with open('prep_test.txt') as f:
        test_data_clean = f.read().splitlines()

# print(test_data_clean[0])
# train_data_clean = preprocess(train_data)
# train_data_clean = remove_stopwords(train_data_clean, stopwords)
# # train_data_clean = stem_words(train_data_clean)
# with open('prep_train.txt', 'w') as f:
#     for item in train_data_clean:
#         f.write("%s\n" % item)
#
# test_data_clean = preprocess(test_data)
# test_data_clean = remove_stopwords(test_data_clean, stopwords)
# test_data_clean = stem_words(test_data_clean)
# with open('prep_test.txt', 'w') as f:
#     for item in test_data_clean:
#         f.write("%s\n" % item)

# print(test_data_clean[0])


# splitting the data
X_train, X_validation, y_train, y_validation = train_test_split(train_data_clean, targets, train_size=0.8, test_size=0.2, random_state=1)

pclf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
#     ('best', TruncatedSVD(n_components=10000)),
    # ('clf', LogisticRegression()),
    ('clf', LinearSVC(C=0.5)),
])


def display_results(y_val, y_pred):
    print(metrics.classification_report(y_val, y_pred))
    print("Accuracy % = ", metrics.accuracy_score(y_val, y_pred))

pclf.fit(X_train, y_train)
y_pred = pclf.predict(test_data_clean)
# print(y_pred)

with open('submission.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("Id", "Category"))
    writer.writerows(zip(test_files_ids, y_pred))

# display_results(y_validation, y_pred)
