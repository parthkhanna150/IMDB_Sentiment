from main import *

SGDpclf = Pipeline([
    ('vect', CountVectorizer(analyzer='word', max_df=1.0, min_df=0,ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', SGDClassifier(alpha=1e-5, epsilon=0.1, random_state=1)),
])

SGDpclf.fit(X_train, y_train)
y_pred_sgd_val = SGDpclf.predict(X_validation)

y_pred_sgd_test = SGDpclf.predict(test_data_clean)
with open('submissionSGD.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("Id", "Category"))
    writer.writerows(zip(test_files_ids, y_pred_sgd_test))

display_results(y_validation, y_pred_sgd_val)

# 91.18%
