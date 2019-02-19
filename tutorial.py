# def load_data():
#     from sklearn.datasets import fetch_20newsgroups
#     newsgroups = fetch_20newsgroups(subset='all')
#     return newsgroups

def tutorial(X, y, n_component=10000):
    import numpy as np
    # X = np.array(X)
    # y = np.array(y)
    print("train test split")
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
    label_train = [e[0] for e in X_train]
    label_valid = [e[0] for e in X_valid]
    X_train = [e[1] for e in X_train]
    X_valid = [e[1] for e in X_valid]

    print("CountVectorizer")
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(ngram_range = (1,2)).fit(X_train)
    X_train_counts = count_vect.transform(X_train)
    X_valid_counts = count_vect.transform(X_valid)
    # print("BinaryVectorizer")
    # binarize_vect = CountVectorizer(ngram_range = (1,2), binary=True).fit(X_train)
    # X_train_binary = binarize_vect.transform(X_train)
    # X_valid_binary = binarize_vect.transform(X_valid)

    print("TfidfTransformer")
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    X_valid_tfidf = tfidf_transformer.transform(X_valid_counts)
    X_train_tfidf= X_train_tfidf.toarray()
    X_valid_tfidf= X_valid_tfidf.toarray()



    print("Normalizer")
    from sklearn.preprocessing import Normalizer
    normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
    X_train_tfidf_norm = normalizer_tranformer.transform(X_train_tfidf)
    X_valid_tfidf_norm = normalizer_tranformer.transform(X_valid_tfidf)
    print(type(X_train_tfidf_norm))

    print("TruncatedSVD")
    import pickle
    from sklearn.decomposition import TruncatedSVD
    svd_transformer = TruncatedSVD(n_components = n_component, random_state=42)
    X_train_tfidf_norm_svd = svd_transformer.transform(X_train_tfidf_norm)
    X_valid_tfidf_norm_svd = svd_transformer.transform(X_valid_tfidf_norm)
    pickle.dump((X_train_tfidf_norm_svd, X_valid_tfidf_norm_svd), open("features_tfidf_norm_svd_%d.cPickle" % n_component, 'wb'))

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    print("Classification: LinearSVC")
    clf = LinearSVC().fit(X_train_tfidf_norm_svd, y_train)
    score = clf.score(X_valid_tfidf_norm_svd, y_valid)
    # clf = LinearSVC().fit(X_train_tfidf_norm, y_train)
    # score = clf.score(X_valid_tfidf_norm, y_valid)
    print("LinearSVC accuracy:", score)

    print("Classification: LogisticRegression")
    clf = LogisticRegression().fit(X_train_tfidf_norm_svd, y_train)
    score = clf.score(X_valid_tfidf_norm_svd, y_valid)
    # clf = LogisticRegression().fit(X_train_tfidf_norm, y_train)
    # score = clf.score(X_valid_tfidf_norm, y_valid)
    print("LinearSVC accuracy:", score)

    # clf = LinearSVC().fit(X_train_tfidf_norm_svd, y_train)
    # score = clf.score(X_valid_tfidf_norm_svd, y_valid)
    # print("LinearSVC accuracy:", score)

    # from sklearn import metrics
    # y_pred = clf.predict(X_valid_tfidf_norm_svd)
    # print(metrics.classification_report(y_valid, y_pred,
    #     target_names=count_vect.get_feature_names()))
    # print("Accuracy", metrics.accuracy_score(y_valid, y_pred))
    #
    # from sklearn.pipeline import Pipeline
    # pclf = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('norm', Normalizer()),
    #     ('clf', MultinomialNB()),
    # ])

    # pclf.fit(X_train, y_train)
    # y_pred = pclf.predict(X_test)
    # print(metrics.classification_report(y_test, y_pred,
    #     target_names=count_vect.get_feature_names()))
    # print(metrics.accuracy_score(y_test, y_pred))

import numpy as np
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def use_pipeline(X, y, n_component=10000):
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
    label_train, label_valid = X_train[:,0], X_test[:,0]
    x_train, X_valid = X_train[:,1:], X_test[:,1:]
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import Normalizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    pclf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('norm', Normalizer()),
        ('clf', MultinomialNB())
        ])
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint as randint
# from scipy.stats import uniform
#
# params = {"vect__ngram_range": [(1,1),(1,2),(2,2)],
#           "tfidf__use_idf": [True, False],
#           "clf__alpha": uniform(1e-2, 1e-3)}
#
# seed = 551 # Very important for repeatibility in experiments!
#
# random_search = RandomizedSearchCV(pclf, param_distributions = params, cv=2, verbose = 10, random_state = seed, n_iter = 1)
# random_search.fit(X_train, y_train)



import pickle
features = pickle.load(open("comment_all.cPickle", "rb"))
target = [0 if i<12500 else 1 for i in range(25000)]
tutorial(X = features, y = target, n_component=10000)

# if features != False:
#     target_vector = np.ones(shape=n_file_pos+n_file_neg)
#     target_vector[n_file_pos:] = 0
#     tutorial(X = features, y = target_vector, n_component=10000)
