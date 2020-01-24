import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

###plots
# dep = 2
# wid = 3
# fig, ax = plt.subplots(dep, wid, sharex=True, sharey=True)
#
# for d in range(dep):
#     for w in range(wid):
#         #ax[d, w].set_title('plot '+ str((d,w)))
#         ax[d, w].boxplot([1,2,2,3,4,5,3,3,2,2,4,5,8,5,5,5], positions=[0])
#         ax[d, w].boxplot([a+2 for a in [1,2,2,3,4,5,3,3,2,2,4,5,8,5,5,5]], positions=[1])
#
# plt.show()

###data generation
seed = 0
n_inf = 10
data, data_labels = sklearn.datasets.make_classification(n_samples=1000,
                                                        n_features=30,
                                                        n_informative=n_inf,
                                                        n_redundant=0, # random linear combinations of informative features
                                                        n_classes=2,
                                                        flip_y=0, # noise
                                                        random_state=seed)
informative_columns = list(range(n_inf))
train_data, test_data, train_labels, test_labels = train_test_split(data, data_labels,
                                                                    test_size = 0.2,
                                                                    random_state = seed)
