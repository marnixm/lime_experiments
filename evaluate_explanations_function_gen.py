import sys
import copy
sys.path.append('..')
import argparse
import explainers
import parzen_windows
import numpy as np
import pickle
import sklearn
from load_datasets import *
from sklearn.metrics import *
#added
import datetime
from explanability_metric import *
import shap
import lime
from scipy import *
from scipy.sparse import *

def get_tree_explanation(tree, v, dataset):
  """ retrieves features used by the tree model iff they appear in instance v """
  t = tree.tree_
  nonzero = v.nonzero()[1 if not dataset=="Generated" else 0]
  current = 0
  left_child = t.children_left[current]
  exp = []
  while left_child != sklearn.tree._tree.TREE_LEAF:
      left_child = t.children_left[current]
      right_child = t.children_right[current]
      f = t.feature[current]
      if f in nonzero:
          exp.append(f)
      if (v[f] if dataset=="Generated" else v[0,f]) < t.threshold[current]:
        current = left_child
      else:
        current = right_child
  return exp
class ExplanationEvaluator:
  def __init__(self, classifier_names=None, logregMaxIter=1000):
    self.classifier_names = classifier_names
    if not self.classifier_names:
      self.classifier_names = ['l1logreg', 'tree']
    self.classifiers = {}
    self.max_iter = logregMaxIter
  def init_classifiers(self, dataset):
    self.classifiers[dataset] = {}
    for classifier in self.classifier_names:
      if classifier == 'l1logreg':
        #print('max iterations logreg', self.max_iter)
        try_cs = np.arange(.1,0,-.01)
        for c in try_cs:
          self.classifiers[dataset]['l1logreg'] = linear_model.LogisticRegression(penalty='l1', fit_intercept=True, C=c,
                                                                                  solver='saga', max_iter=self.max_iter)
          self.classifiers[dataset]['l1logreg'].fit(self.train_vectors[dataset], self.train_labels[dataset])

          #maximum number of features logreg uses for any instance is K=10
          # coef = self.classifiers[dataset]['l1logreg'].coef_[0]
          # # coefNonZero = coef.nonzero()[0]
          # # nonzero = np.split(self.train_vectors[dataset].indices, self.train_vectors[dataset].indptr[1:-1])
          # lengths = [len(coefNonZero) for row in nonzero]
          length = len(self.classifiers[dataset]['l1logreg'].coef_[0].nonzero()[0])

          if np.average(length) <= 10:
            print('Logreg for ', dataset, ' has length',  np.mean(length), 'with C=', c)
            #print('And max length = ', np.max(lengths), ', min length = ', np.min(lengths))
            break
      if classifier == 'tree':
        self.classifiers[dataset]['tree'] = tree.DecisionTreeClassifier(random_state=1)
        self.classifiers[dataset]['tree'].fit(self.train_vectors[dataset], self.train_labels[dataset])
        lengths = [len(get_tree_explanation(self.classifiers[dataset]['tree'], self.train_vectors[dataset][i], dataset)) for i in range(self.train_vectors[dataset].shape[0])]
        """f = set(self.classifiers[dataset]['tree'].tree_.feature) #features of model
        lengths = [len(np.intersect1d(f, set(self.train_vectors[dataset][i].nonzero()[1]))) for i in range (self.train_vectors[dataset].shape[0])]
        """ #todo^^
        print('Tree for ', dataset, ' has mean length',  np.mean(lengths))
  def load_datasets(self, dataset_names, parameters):
    self.train_data = {}
    self.train_labels = {}
    self.test_data = {}
    self.test_labels = {}
    for dataset in dataset_names:
      self.train_data[dataset], self.train_labels[dataset], self.test_data[dataset], self.test_labels[dataset], _ = LoadDataset(dataset, parameters)
  def vectorize_and_train(self, dataset_names):
    self.vectorizer = {}
    self.train_vectors = {}
    self.test_vectors = {}
    self.inverse_vocabulary = {}
    # print('Vectorizing...')
    for d in self.train_data:
      if list(self.train_data.keys())[0]=='Generated':
        self.train_vectors[d] = self.train_data[d]
        self.test_vectors[d] = self.test_data[d]
      else:
        self.vectorizer[d] = CountVectorizer(lowercase=False, binary=True)
        self.train_vectors[d] = self.vectorizer[d].fit_transform(self.train_data[d])
        self.test_vectors[d] = self.vectorizer[d].transform(self.test_data[d])
        terms = np.array(list(self.vectorizer[d].vocabulary_.keys()))
        indices = np.array(list(self.vectorizer[d].vocabulary_.values()))
        self.inverse_vocabulary[d] = terms[np.argsort(indices)]
    # print('Done')
    # print('Training...')
    for d in self.train_data:
      print(d)
      self.init_classifiers(d)
    # print('Done')
    # print()
  def measure_explanation_hability(self, explain_fn, max_examples=None):
    """Asks for explanations for all predictions in the train and test set, with
    budget = size of explanation. Returns two maps (train_results,
    test_results), from dataset to classifier to list of recalls"""
    budget = 10
    train_results = {}
    test_results = {}
    faith = {}
    ndcg = {}
    for d in self.train_data:
      train_results[d] = {}
      test_results[d] = {}
      faith[d] = {}
      ndcg[d] = {}
      print('Dataset:', d)
      for c in self.classifiers[d]:
        train_results[d][c] = []
        test_results[d][c] = []
        faith[d][c] = []
        ndcg[d][c] = []
        if c == 'l1logreg':
          c_features = self.classifiers[d][c].coef_.nonzero()
          c_importance = self.classifiers[d][c].coef_[c_features]
          c_features = c_features[1] #weird dimentions
        if c == 'tree':
          c_features = self.classifiers[d][c].feature_importances_.nonzero()
          c_importance = self.classifiers[d][c].feature_importances_[c_features]
          c_features = c_features[0] #weird dimentions
        ###order all model features by absolute feature importance
        c_features = [f for _, f in sorted(zip(c_importance,c_features), key=lambda z: abs(z[0]), reverse=True)]
        print('classifier:', c)
        for i in range(len(self.test_data[d])):
          if d == "Generated":  df = list(range(len(self.test_vectors[d][i]))) #all features exist for generated data
          elif c == 'l1logreg': df = self.test_vectors[d][i].nonzero()[1]
          elif c == 'tree':     df = get_tree_explanation(self.classifiers[d][c], self.test_vectors[d][i], d)
          ###order by feature importance
          true_features = [f for _, f in sorted(zip(c_features, df), key=lambda z: z[0], reverse=True)]
          if len(true_features) == 0:
            continue
          """if len(df) > 2 and not np.array_equal(df,true_features) and c =='tree':
            print(df, true_features)"""

          exp = explain_fn(self.test_vectors[d][i], self.test_labels[d][i], self.classifiers[d][c], budget, d)
          exp_features = [x[0] for x in exp]
          #true_features = true_features[:budget] #cut-off at budget=10
          #todo check impact of below
          #exp_features = exp_features[:len(true_features)] #cut-off at |tf|<=10
          #Recall
          test_results[d][c].append(float(len(np.intersect1d(true_features, exp_features))
                                          / len(true_features)))
          #Faithfulness
          # faith[d][c].append(faithfulness(exp, self.classifiers[d][c], self.test_vectors[d][i]))
          #NDCG
          # if len(exp_features) < len(true_features): #todo test
          #   print('true',true_features)
          #   print('exp', exp_features)
          #   a=2
          true_features = true_features[:len(exp_features)]  # cut-off at |ef|<=10 todo
          exp_features = exp_features[:len(true_features)]  # cut-off at |tf|<=10
          ndcg[d][c].append(ndcg_score(true_features, exp_features))
          if max_examples and i >= max_examples:
            break
    return train_results, test_results, faith, ndcg

def main(dataset, algorithm, explain_method, parameters):
  startTime = datetime.datetime.now()
  path = os.path.abspath(os.curdir) + '/log_5.2/' + \
         str(startTime.strftime('%y%m%d %H.%M.%S')) \
         + ' ' + dataset[-5:] + ' ' + algorithm + ' ' + explain_method +'.txt'
  print('Start', datetime.datetime.now().strftime('%H.%M.%S'))

  evaluator = ExplanationEvaluator(classifier_names=[algorithm], logregMaxIter=parameters['max_iter_logreg'])
  evaluator.load_datasets([dataset], parameters)
  evaluator.vectorize_and_train([dataset])
  explain_fn = None
  if explain_method == 'lime':
    rho, num_samples = parameters['lime']['rho'], parameters['lime']['num_samples']
    if dataset=="Generated":
      explainer = explainers.LimeTabExplainer(evaluator.train_vectors[dataset], nsamples=num_samples, K=parameters['max_examples'])
    else:
      kernel = lambda d: np.sqrt(np.exp(-(d ** 2) / rho ** 2))
      explainer = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text, num_samples=num_samples,
                                                       return_mean=False, verbose=False, return_mapped=True)
    explain_fn = explainer.explain_instance
  elif explain_method == 'parzen':
    sigmas = {'multi_polarity_electronics': {'tree': 0.5, 'l1logreg': 1},
    'multi_polarity_kitchen': {'tree': 0.75, 'l1logreg': 2.0},
    'multi_polarity_dvd': {'tree': 8.0, 'l1logreg': 1},
    'multi_polarity_books': {'tree': 2.0, 'l1logreg': 2.0},
    'Generated': {'tree': 2.0, 'l1logreg': 2.0}} #todo optimize?
    explainer = parzen_windows.ParzenWindowClassifier()
    cv_preds = sklearn.model_selection.cross_val_predict(evaluator.classifiers[dataset][algorithm], evaluator.train_vectors[dataset],
                                                         evaluator.train_labels[dataset], cv=parameters['parzen_num_cv'])
    explainer.fit(evaluator.train_vectors[dataset], cv_preds, dataset)
    explainer.sigma = sigmas[dataset][algorithm]
    explain_fn = explainer.explain_instance
  # greedy/random cannot be score by faithfullness measure
  # elif explain_method == 'greedy':
  #  explain_fn = explainers.explain_greedy
  # elif explain_method == 'random':
  #  explainer = explainers.RandomExplainer()
  #  explain_fn = explainer.explain_instance
  elif explain_method == 'shap':
    nsamples, num_features, n_clusters = parameters['shap']['nsamples'], parameters['shap']['num_features'], parameters['shap']['n_clusters']
    explainer = explainers.ShapExplainer(evaluator.classifiers[dataset][algorithm], evaluator.train_vectors[dataset],
                                         nsamples=nsamples, num_features=num_features,
                                         num_clusters=(n_clusters if dataset=="Generated" else None))
    explain_fn = explainer.explain_instance

  train_results, test_results, faith, ndcg = evaluator.measure_explanation_hability(explain_fn, max_examples=parameters['max_examples'])
  #print results
  print('Finish', datetime.datetime.now().strftime('%H.%M.%S'))
  print('Calc time',round((datetime.datetime.now()-startTime).total_seconds()/60,3),' min\n\n')
  print('Average test: ', np.mean(test_results[dataset][algorithm]))
  out = {'train': train_results[dataset][algorithm], 'test' : test_results[dataset][algorithm]}
  return {'dataset': dataset, 'alg': algorithm, 'exp':  explain_method,
          'score':  test_results[dataset][algorithm],
          'faithfulness': faith[dataset][algorithm],
          'ndcg': ndcg[dataset][algorithm],
          'calcTime': round((datetime.datetime.now()-startTime).total_seconds()/60,3)}

if __name__ == "__main__":
    main()
