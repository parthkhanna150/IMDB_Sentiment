from main import *
import numpy as np
from sklearn.model_selection import KFold

def cross_validation(X_train, y_train, X_test, n_fold=5, max_df = 0.2):
    X = X_train
    y = np.array(y_train)
    result_valid = np.zeros(shape=(len(X_train)))
    result_test = np.zeros(shape=(len(X_test), n_fold))

    kf = KFold(n_splits = n_fold, shuffle = True, random_state=1)
    fold_idx = 0
    for train_index, valid_index in kf.split(X):
        print("fold_idx", fold_idx)
        X_train = np.array([X[i] for i in train_index])
        X_valid = np.array([X[i] for i in valid_index])
        y_train, y_valid = y[train_index], y[valid_index]

        pclf = Pipeline([
            ('vect', CountVectorizer(analyzer='word', max_df=1, min_df=0,ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer(norm='l2')),
            ('clf', SGDClassifier(alpha=1e-5, epsilon=0.1, random_state=1)),
        ])

        pclf.fit(X_train, y_train)
        y_pred_val = pclf.predict(X_valid)
        result_valid[valid_index] = y_pred_val
        y_pred_test = pclf.predict(test_data_clean)
        result_test[:, fold_idx] = y_pred_test
        fold_idx += 1
        display_results(y_valid, y_pred_val)

    print("Accuracy on entire validation")
    display_results(y, result_valid)
    y_pred_test = np.array([1 if result_test[sample_idx, :].sum() > 1 else 0 for sample_idx in range(result_test.shape[0])])

    with open('submission_cross_validation.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("Id", "Category"))
        writer.writerows(zip(test_files_ids, y_pred_test))


cross_validation(X_train = train_data_clean, y_train = targets, X_test = test_data_clean, n_fold = 5, max_df = 0.5)
