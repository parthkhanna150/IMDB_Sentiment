from main import *
from sgd import sgd
from svm import svm
from log_reg import log_reg
import time
import pickle
import os
import pandas as pd
from plot_tools import clf_time_plot

n_iter = 10

if not os.path.isfile("clf_time_iter_%d.cPickle" % n_iter):

    sgd_time_list = []
    svm_time_list = []
    logreg_time_list = []
    print("Generating run_time values")
    for i in range(n_iter):

        print("Iteration", i)
        X_train, X_valid, y_train, y_valid = train_test_split(train_data_clean, targets, train_size=0.8, test_size=0.2, shuffle=True)

        time_init = time.time()
        var = sgd(max_df = 0.5, reset_train_test_split = True, X_train_arg = X_train, y_train_arg = y_train, X_valid_arg = X_valid, y_valid_arg = y_valid)
        time_diff = time.time() - time_init
        sgd_time_list.append(time_diff)

        time_init = time.time()
        var = svm(max_df = 0.5, reset_train_test_split = True, X_train_arg = X_train, y_train_arg = y_train, X_valid_arg = X_valid, y_valid_arg = y_valid)
        time_diff = time.time() - time_init
        svm_time_list.append(time_diff)

        time_init = time.time()
        var = log_reg(max_df = 0.5, reset_train_test_split = True, X_train_arg = X_train, y_train_arg = y_train, X_valid_arg = X_valid, y_valid_arg = y_valid)
        time_diff = time.time() - time_init
        logreg_time_list.append(time_diff)

        df = pd.DataFrame(columns=['clf', 'run_time'])
        row_idx = 0
        for sgd_val in sgd_time_list:
            s = pd.Series(data={'clf':'SGD', 'run_time':sgd_val})
            df.loc[row_idx] = s
            row_idx += 1
        for svm_val in svm_time_list:
            s = pd.Series(data={'clf':'SVM', 'run_time':svm_val})
            df.loc[row_idx] = s
            row_idx += 1
        for logreg_val in logreg_time_list:
            s = pd.Series(data={'clf':'LR', 'run_time':logreg_val})
            df.loc[row_idx] = s
            row_idx += 1
    pickle.dump(df, open("clf_time_iter_%d.cPickle" % n_iter, 'wb'))
    print("Saved to file\nclf_time_iter_%d.cPickle" % n_iter)
else:
    df = pickle.load(open("clf_time_iter_%d.cPickle" % n_iter, 'rb'))
    print("Read from file\nclf_time_iter_%d.cPickle" % n_iter)
# sgd_time_list, svm_time_list, logreg_time_list = pickle.load(open("clf_time_iter_%d.cPickle" % n_iter, 'rb'))
clf_time_plot(df, n_iter)
