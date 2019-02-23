import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def max_df_plot(max_df_list, max_df_logreg, max_df_svm, max_df_sgd):
    plt.plot(max_df_list, max_df_logreg, c="g", alpha=0.5, marker='o', label="LR")
    plt.plot(max_df_list, max_df_svm, c="r", alpha=0.5, marker='*', label="SVM")
    plt.plot(max_df_list, max_df_sgd, c="b", alpha=0.5, marker='^', label="SGD")
    plt.title("Wordstop")
    plt.xlabel("max_df")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("max_df_%d.png" % len(max_df_list))

# max_df_list = [0.5, 1.0]
# max_df_logreg = [0.88, 0.78]
# max_df_svm = [0.67, 0.85]
# max_df_sgd = [0.73, 0.79]
#
# max_df_plot(max_df_list, max_df_logreg, max_df_svm, max_df_sgd)
def max_df_plot(max_df_list, max_df_logreg, max_df_svm, max_df_sgd):
    plt.plot(max_df_list, max_df_logreg, c="g", alpha=0.5, marker='o', label="LR")
    plt.plot(max_df_list, max_df_svm, c="r", alpha=0.5, marker='*', label="SVM")
    plt.plot(max_df_list, max_df_sgd, c="b", alpha=0.5, marker='^', label="SGD")
    plt.title("Wordstop")
    plt.xlabel("max_df")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("max_df_%d.png" % len(max_df_list))

def clf_time_plot(df, n_iter):
    plt.clf()
    sns.set(style="whitegrid")
    f, ax = plt.subplots()
    sns.barplot(x="clf", y="run_time", data = df, hue = "clf", palette = "Set3", which='major',showmeans=True)
    plt.ylabel("run time (s)")
    plt.title("Runtime of LinearSVC vs. SGDClassifier vs. LogisticRegression\n2 Categories, %d Runs" % n_iter)
    plt.legend(loc="lower left")
    plt.savefig("clf_runtime_iter_%d.png" % n_iter)

def sketch_distribution(s1, s2, n_iter):
    s1 = np.array(s1)
    s2 = np.array(s2)
    n_iter = len(s1)
    fileName = 'unigram_vs_bigram_iter_%d.png' % n_iter
    plt.clf()
    title = "Accuracy Distribution of Unigram vs. Uni/Bigram\nClassification with LinearSVC, 2 Categories, %d Runs Discretized" % n_iter
    f, ax = plt.subplots()
    sns.distplot(a = s1, label = 'Uni/Bigram (mean %.3f' % s1.mean() + ", stD %.3f" % s1.std() +')', rug = True, norm_hist = True)
    sns.distplot(a = s2, label = 'Unigram (mean %.3f' % s2.mean() + ", stD %.3f" % s2.std() + ')', rug = True, norm_hist = True)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Accuracy")
    ax.legend(ncol=1, loc="best", frameon=True)
    plt.savefig(fileName)

def sketch_distribution_3models(sgd_accuracy_list, svm_accuracy_list, lr_accuracy_list, n_iter):
    s1 = np.array(sgd_accuracy_list)
    s2 = np.array(svm_accuracy_list)
    s3 = np.array(lr_accuracy_list)
    print(s1.shape, s2.shape, s3.shape)
    n_iter = len(s1)
    fileName = 'sdg_vs_svm_vs_lr_iter_%d.png' % n_iter
    plt.clf()
    title = "Accuracy Distribution of SGDClassifier vs. LinearSVC vs. LogisticRegression\n2 Categories, %d Runs Discretized" % n_iter
    sns.set(style="whitegrid")
    f, ax = plt.subplots()
    sns.distplot(a = s1, label = 'sgd (mean %.3f' % s1.mean() + ", stD %.3f" % s1.std() +')', rug = True, norm_hist = True)
    sns.distplot(a = s2, label = 'svm (mean %.3f' % s2.mean() + ", stD %.3f" % s2.std() + ')', rug = True, norm_hist = True)
    sns.distplot(a = s3, label = 'lr (mean %.3f' % s3.mean() + ", stD %.3f" % s3.std() + ')', rug = True, norm_hist = True)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Accuracy")
    ax.legend(ncol=1, loc="best", frameon=True)
    plt.savefig(fileName)
