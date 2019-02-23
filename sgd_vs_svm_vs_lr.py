from main import *
from sgd import sgd
from svm import svm
from log_reg import log_reg
import time
import pickle
import os
import pandas as pd
from plot_tools import sketch_distribution_3models

n_iter = 2

if not os.path.isfile("sgd_vs_svm_vs_lr_iter_%d.cPickle" % n_iter):

    sgd_accuracy_list = []
    svm_accuracy_list = []
    logreg_accuracy_list = []
    print("Generating accoracy values")
    for i in range(n_iter):

        print("Iteration", i)
        X_train, X_valid, y_train, y_valid = train_test_split(train_data_clean, targets, train_size=0.8, test_size=0.2, shuffle=True)

        var = sgd(max_df = 0.5, reset_train_test_split = True, X_train_arg = X_train, y_train_arg = y_train, X_valid_arg = X_valid, y_valid_arg = y_valid)
        sgd_accuracy_list.append(var)

        var = svm(max_df = 0.5, reset_train_test_split = True, X_train_arg = X_train, y_train_arg = y_train, X_valid_arg = X_valid, y_valid_arg = y_valid)
        svm_accuracy_list.append(var)

        var = log_reg(max_df = 0.5, reset_train_test_split = True, X_train_arg = X_train, y_train_arg = y_train, X_valid_arg = X_valid, y_valid_arg = y_valid)
        logreg_accuracy_list.append(var)

    pickle.dump((sgd_accuracy_list, svm_accuracy_list, logreg_accuracy_list), open("sgd_vs_svm_vs_lr_iter_%d.cPickle" % n_iter, 'wb'))
    print("Saved to file\nsgd_vs_svm_vs_lr_iter_%d.cPickle" % n_iter)
else:
    sgd_accuracy_list, svm_accuracy_list, logreg_accuracy_list = pickle.load(open("sgd_vs_svm_vs_lr_iter_%d.cPickle" % n_iter, 'rb'))
    print("Read from file\nsgd_vs_svm_vs_lr_iter_%d.cPickle" % n_iter)
# sgd_time_list, svm_time_list, logreg_time_list = pickle.load(open("clf_time_iter_%d.cPickle" % n_iter, 'rb'))
sketch_distribution_3models(sgd_accuracy_list, svm_accuracy_list, logreg_accuracy_list, n_iter)
