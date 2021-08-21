import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_single_dist(x):
    sns.distplot(x)
    plt.show()

def plot_overlap_dist(x, y):
    sns.distplot(x)
    sns.distplot(y)
    plt.show()


# tester
#batch = 1000
#x = np.random.rand(batch,1)
#plot_single_dist(x)
#y_baseline = np.random.rand(batch,1)
#plot_overlap_dist(x, y_baseline)

