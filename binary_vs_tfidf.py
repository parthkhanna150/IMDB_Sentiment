from main import *
from plot_tools import sketch_distribution
import pickle
import os

def svm(max_df=0.5):
    binary_SVMpclf = Pipeline([
        ('vect', CountVectorizer(max_df=max_df, binary=True, ngram_range=(1,2))),
        ('clf', LinearSVC(C=0.5)),
    ])

    binary_SVMpclf.fit(X_train, y_train)
    binary_score = binary_SVMpclf.score(X_valid, y_valid)

    tfidf_SVMpclf = Pipeline([
        ('vect', CountVectorizer(max_df=max_df, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', LinearSVC(C=0.5)),
    ])

    tfidf_SVMpclf.fit(X_train, y_train)
    tfidf_score = tfidf_SVMpclf.score(X_valid, y_valid)

    return binary_score, tfidf_score

n_iter = 1000

if not os.path.isfile("TFIDF-Binary_score_list_iter_%d.cPickle" % n_iter):
    binary_score_list = []
    tfidf_score_list = []
    for i in range(n_iter):
        print("Iteration", i)
        X_train, X_valid, y_train, y_valid = train_test_split(train_data_clean, targets, train_size=0.8, test_size=0.2, shuffle=True)
        binary_score, tfidf_score = svm()
        binary_score_list.append(binary_score)
        tfidf_score_list.append(tfidf_score)
    pickle.dump((tfidf_score_list, binary_score_list), open("TFIDF-Binary_score_list_iter_%d.cPickle" % n_iter, 'wb'))
    print("Saved to file: %s" % "TFIDF-Binary_score_list_iter_%d.cPickle" % n_iter)
else:
    tfidf_score_list, binary_score_list = pickle.load(open("TFIDF-Binary_score_list_iter_%d.cPickle" % n_iter, 'rb'))
    print("Read from file: %s" % "TFIDF-Binary_score_list_iter_%d.cPickle" % n_iter)
sketch_distribution(s1 = tfidf_score_list, s2 = binary_score_list, n_iter = n_iter)
