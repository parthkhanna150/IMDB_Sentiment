from plot_tools import max_df_plot
from log_reg import log_reg
from svm import svm
from sgd import sgd
import pickle
import os
import numpy as np

max_df_list = np.linspace(0.05, 1, 20)

if not os.path.isfile("max_df_%d.cPickle" % len(max_df_list)):
    print("Generate ")
    max_df_logreg = []
    max_df_svm = []
    max_df_sgd = []
    counter = 0
    for max_df in max_df_list:
        print("Iteration", counter)
        max_df_logreg.append(log_reg(max_df))
        max_df_svm.append(svm(max_df))
        max_df_sgd.append(sgd(max_df))
        counter += 1
    pickle.dump((max_df_list, max_df_logreg, max_df_svm, max_df_sgd), open("max_df_%d.cPickle" % len(max_df_list), 'wb'))
else:
    max_df_list, max_df_logreg, max_df_svm, max_df_sgd = pickle.load(open("max_df_%d.cPickle" % len(max_df_list), 'rb'))

max_df_plot(max_df_list, max_df_logreg, max_df_svm, max_df_sgd)
