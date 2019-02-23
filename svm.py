from main import *

SVMpclf = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', LinearSVC(C=0.5)),
])

SVMpclf.fit(X_train, y_train)
y_pred_svm_val = SVMpclf.predict(X_validation)
y_pred_svm_test = SVMpclf.predict(test_data_clean)
with open('submissionLSVM.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("Id", "Category"))
    writer.writerows(zip(test_files_ids, y_pred_svm_test))

display_results(y_validation, y_pred_svm_val)

# 91.08%
