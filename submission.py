import pandas as pd
from svm import y_pred_svm_test
from sgd import y_pred_sgd_test
from log_reg import y_pred_lr_test

preds_val = pd.DataFrame()
preds_val['pred_val_1'] = y_pred_sgd_val
preds_val['pred_val_2'] = y_pred_svm_val
preds_val['pred_val_3'] = y_pred_lr_val

preds_val['maj_vote_val'] = preds_val.mode(axis = 1)

with open('submission.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("Id", "Category"))
    writer.writerows(zip(test_files_ids, preds['maj_vote']))
