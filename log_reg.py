from main import *
LRpclf = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', LogisticRegression()),
])

LRpclf.fit(X_train, y_train)
y_pred_lr_val = LRpclf.predict(X_validation)
y_pred_lr_test = LRpclf.predict(test_data_clean)
with open('submissionLR.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("Id", "Category"))
    writer.writerows(zip(test_files_ids, y_pred_lr_test))

display_results(y_validation, y_pred_lr_val)

# 89.66%
