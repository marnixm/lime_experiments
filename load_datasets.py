import random
import os
import re
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
# PUT POLARITY DATASET PATH HERE
POLARITY_PATH = os.path.abspath(os.curdir) + '\\Data\\multi_polarity_books\\'

def LoadDataset(dataset_name, parameters):
  if dataset_name.endswith('ng'):
    if dataset_name == '2ng':
      cats = ['alt.atheism', 'soc.religion.christian']
      class_names = ['Atheism', 'Christianity']
    if dataset_name == 'talkng':
      cats = ['talk.politics.guns', 'talk.politics.misc']
      class_names = ['Guns', 'PoliticalMisc']
    if dataset_name == '3ng':
      cats = ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.windows.x']
      class_names = ['windows.misc', 'ibm.hardware', 'windows.x']
    newsgroups_train = fetch_20newsgroups(subset='train',categories=cats)
    newsgroups_test = fetch_20newsgroups(subset='test',categories=cats)
    train_data = newsgroups_train.data
    train_labels = newsgroups_train.target
    test_data = newsgroups_test.data
    test_labels = newsgroups_test.target
    return train_data, train_labels, test_data, test_labels, class_names
  if dataset_name.startswith('multi_polarity'):
    name = dataset_name.split('_')[2]
    return LoadMultiDomainDataset(POLARITY_PATH + name)
  if dataset_name.find('generated'):
    # todo implement generated data
    parameters = parameters['Gen']
    data, data_labels = sklearn.datasets.make_classification(n_samples=parameters['nrows'],
                                                             n_features=parameters['n_features'],
                                                             n_informative=parameters['n_inf'],
                                                             n_redundant=0, # random linear combinations of informative features
                                                             n_classes=2,
                                                             flip_y=parameters['noise'],
                                                             random_state=parameters['seed'])
    # informative_columns = list(range(n_inf))
    train_data, test_data, train_labels, test_labels = train_test_split(data, data_labels,
                                                                        test_size = 0.2,
                                                                        random_state = parameters['seed'])
    return train_data, np.array(train_labels), test_data, np.array(test_labels), ""

def LoadMultiDomainDataset(path_data, remove_bigrams=True):
  random.seed(1)
  pos = []
  neg = []
  def get_words(line, remove_bigrams=True):
    z = [tuple(x.split(':')) for x in re.findall('\w*?:\d', line)]
    if remove_bigrams:
      z = ' '.join([' '.join([x[0]] * int(x[1])) for x in z if '_' not in x[0]])
    else:
      z = ' '.join([' '.join([x[0]] * int(x[1])) for x in z])
    return z
  for line in open(os.path.join(path_data, 'negative.review')):
    neg.append(get_words(line, remove_bigrams))
  for line in open(os.path.join(path_data, 'positive.review')):
    pos.append(get_words(line, remove_bigrams))
  random.shuffle(pos)
  random.shuffle(neg)
  split_pos = int(len(pos) * .8)
  split_neg = int(len(neg) * .8)
  train_data = pos[:split_pos] + neg[:split_neg]
  test_data = pos[split_pos:] + neg[split_neg:]
  train_labels = [1] * len(pos[:split_pos]) + [0] * len(neg[:split_neg])
  test_labels = [1] * len(pos[split_pos:]) + [0] * len(neg[split_neg:])
  return train_data, np.array(train_labels), test_data, np.array(test_labels), ['neg', 'pos']
