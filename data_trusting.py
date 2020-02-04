import sys
import copy
import os
import numpy as np
import scipy as sp
import json
import random
import sklearn
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
import pickle
import explainers
import parzen_windows
import embedding_forest
from load_datasets import *
import argparse
import collections
import datetime
from explanability_metric import *

def get_classifier(name, vectorizer, parameters):
  if name == 'logreg':
    return linear_model.LogisticRegression(fit_intercept=True, solver='lbfgs')
  if name == 'random_forest':
    return ensemble.RandomForestClassifier(n_estimators=parameters['rf']['n_estimators'], random_state=1, max_depth=5, n_jobs=10)
  if name == 'svm':
    return svm.SVC(probability=True, kernel='rbf', C=10, gamma=0.001)
  if name == 'tree':
    return tree.DecisionTreeClassifier(random_state=1)
  if name == 'neighbors':
    return neighbors.KNeighborsClassifier()
  if name == 'embforest':
    return embedding_forest.EmbeddingForest(vectorizer)

def main(dataset, algorithm, parameters):
  num_features = parameters['num_features']
  percent_untrustworthy = parameters['percent_untrustworthy']
  num_rounds = parameters['num_rounds']
  max_examples = parameters['max_examples']
  test_against = parameters['test_against']

  startTime = datetime.datetime.now()
  print('Start', datetime.datetime.now().strftime('%H.%M.%S'))

  train_data, train_labels, test_data, test_labels, class_names, perturb_instance = LoadDataset(dataset, parameters)
  vectorizer = CountVectorizer(lowercase=False, binary=True)
  if dataset == 'Generated':
    train_vectors = train_data
    test_vectors = test_data
  else:
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)
    terms = np.array(list(vectorizer.vocabulary_.keys()))
    indices = np.array(list(vectorizer.vocabulary_.values()))
    inverse_vocabulary = terms[np.argsort(indices)]

  np.random.seed(1)
  classifier = get_classifier(algorithm, vectorizer, parameters)
  classifier.fit(train_vectors, train_labels)
  predictions = classifier.predict(test_vectors)
  predict_probas = classifier.predict_proba(test_vectors)[:, 1] #predict/explain positive instances

  untrustworthy_rounds = []
  all_features = list(range(train_vectors.shape[1]))
  num_untrustworthy = int(train_vectors.shape[1] * percent_untrustworthy)
  for _ in range(num_rounds):
    untrustworthy_rounds.append(np.random.choice(all_features, num_untrustworthy, replace=False))

  rho, num_samples = parameters['lime']['rho'], parameters['lime']['num_samples']
  if dataset=="Generated":
    #tabular explainer for generated data
    LIME = explainers.LimeTabExplainer(train_vectors, nsamples=num_samples, K=parameters['max_examples'], return_mean=True)
  else:
    kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
    LIME = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text,
                                                num_samples=num_samples, return_mean=True, return_mapped=True)
  nsamples, n_clusters = parameters['shap']['nsamples'], parameters['shap']['n_clusters']
  SHAP = explainers.ShapExplainer(classifier, train_vectors, nsamples=nsamples,
                                  num_features=parameters['shap']['num_features'],
                                  num_clusters=(n_clusters if dataset=="Generated" else None))

  parzen = parzen_windows.ParzenWindowClassifier()
  cv_preds = sklearn.model_selection.cross_val_predict(classifier, train_vectors, train_labels, cv=parameters['parzen_num_cv'])
  parzen.fit(train_vectors, cv_preds, dataset)
  sigmas = {'multi_polarity_electronics': {'neighbors': 0.75, 'svm': 10.0, 'tree': 0.5, 'logreg': 0.5, 'random_forest': 0.5, 'embforest': 0.75},
  'multi_polarity_kitchen':               {'neighbors': 1.0, 'svm': 6.0, 'tree': 0.75, 'logreg': 0.25, 'random_forest': 6.0, 'embforest': 1.0},
  'multi_polarity_dvd':                   {'neighbors': 0.5, 'svm': 0.75, 'tree': 8.0, 'logreg': 0.75, 'random_forest': 0.5, 'embforest': 5.0},
  'multi_polarity_books':                 {'neighbors': 0.5, 'svm': 7.0, 'tree': 2.0, 'logreg': 1.0, 'random_forest': 1.0, 'embforest': 3.0},
  'Generated':                            {'neighbors': 0.5, 'svm': 7.0, 'tree': 2.0, 'logreg': 1.0, 'random_forest': 1.0, 'embforest': 3.0}}
  parzen.sigma = sigmas[dataset][algorithm]

  exps = {}
  diff = {}
  accuracy = {}
  explainer_names = ['shap', 'lime','parzen']
  trust_fn = lambda prev, curr: ((prev > 0.5 and curr > 0.5) or (prev <= 0.5 and curr <= 0.5))
  trust_fn_all = lambda exp, unt: len([x[0] for x in exp if x[0] in unt]) == 0
  for expl in explainer_names: exps[expl] = []
  for expl in explainer_names[:3]: diff[expl] = []
  for expl in explainer_names: accuracy[expl] = []

  for i in range(test_vectors.shape[0]):
    if i==max_examples: break
    if i%100==0: print(i,'/',test_vectors.shape[0])
    sys.stdout.flush()
    exp, mean = LIME.explain_instance(test_vectors[i], 1, classifier.predict_proba, num_features)
    accuracy['lime'].append(trust_fn(predict_probas[i], mean + sum([x[1] for x in exp])))
    exps['lime'].append((exp, mean))

    exp = SHAP.explain_instance(test_vectors[i], None, None, None, dataset, predictPositive=True)
    accuracy['shap'].append(trust_fn(predict_probas[i],
                                     SHAP.explainer.expected_value[1] + sum([x[1] for x in exp])))
    exps['shap'].append(exp)

    exp = parzen.explain_instance(test_vectors[i], 1, classifier.predict_proba, num_features, dataset)
    mean = parzen.predict_proba(test_vectors[i])[1]
    accuracy['parzen'].append(trust_fn(predict_probas[i],
                                     mean + sum([x[1] for x in exp])))
    exps['parzen'].append((exp, mean))

  precision = {}
  recall = {}
  f1 = {}
  for name in explainer_names:
    precision[name] = []
    recall[name] = []
    f1[name] = []
  flipped_preds_size = []
  for untrustworthy in untrustworthy_rounds:
    t = test_vectors.copy()
    t[:, untrustworthy] = 0
    mistrust_idx = np.argwhere(classifier.predict(t) != classifier.predict(test_vectors)).flatten()
    #print('Number of suspect predictions', len(mistrust_idx))
    shouldnt_trust = set(mistrust_idx)
    flipped_preds_size.append(len(shouldnt_trust))

    mistrust = collections.defaultdict(lambda:set())
    trust = collections.defaultdict(lambda: set())
    for i in range(test_vectors.shape[0]):
      if i==max_examples: break
      exp, mean = exps['lime'][i]
      prev_tot = predict_probas[i]
      prev_tot2 = sum([x[1] for x in exp]) + mean #what it should have been
      tot = prev_tot2 - sum([x[1] for x in exp if x[0] in untrustworthy]) #discounted effect
      trust['lime'].add(i) if trust_fn(tot, prev_tot2) else mistrust['lime'].add(i)

      exp = exps['shap'][i]
      prev_tot = predict_probas[i]
      prev_tot2 = SHAP.explainer.expected_value[1] + sum([x[1] for x in exp])
      tot = prev_tot2 - sum([x[1] for x in exp if x[0] in untrustworthy])
      trust['shap'].add(i) if trust_fn(tot, prev_tot2) else mistrust['shap'].add(i)

      exp, mean = exps['parzen'][i]
      prev_tot = mean
      tot = mean - sum([x[1] for x in exp if x[0] in untrustworthy])
      trust['parzen'].add(i) if trust_fn(tot, prev_tot) else mistrust['parzen'].add(i)

    for expl in explainer_names:
      false_positives = set(trust[expl]).intersection(shouldnt_trust)
      true_positives = set(trust[expl]).difference(shouldnt_trust)
      false_negatives = set(mistrust[expl]).difference(shouldnt_trust)
      true_negatives = set(mistrust[expl]).intersection(shouldnt_trust)

      try:
        prec= len(true_positives) / float(len(true_positives) + len(false_positives))
      except:
        prec= 0
      try:
        rec= float(len(true_positives)) / (len(true_positives) + len(false_negatives))
      except:
        rec= 0
      precision[expl].append(prec)
      recall[expl].append(rec)
      f1z = 2 * (prec * rec) / (prec + rec) if (prec and rec) else 0
      f1[expl].append(f1z)

  #print results
  print('Finish', datetime.datetime.now().strftime('%H.%M.%S'))
  print('Calc time',round((datetime.datetime.now()-startTime).total_seconds()/60,3),' min\n\n')
  print('Average number of flipped predictions:', np.mean(flipped_preds_size), '+-', np.std(flipped_preds_size), '\n')
  print('Precision:')
  results = {'Precision': {}, 'Recall': {}, 'F1': {}}
  for expl in explainer_names:
    print(expl, np.mean(precision[expl]), '+-', np.std(precision[expl]), 'pvalue', sp.stats.ttest_ind(precision[expl], precision[test_against])[1].round(4))
    inf = [np.mean(precision[expl]), np.std(precision[expl]), sp.stats.ttest_ind(precision[expl], precision[test_against])[1].round(4)]
    results['Precision'].update({expl: inf})
  print('')
  print('Recall:')
  for expl in explainer_names:
    print(expl, np.mean(recall[expl]), '+-', np.std(recall[expl]), 'pvalue', sp.stats.ttest_ind(recall[expl], recall[test_against])[1].round(4))
    results['Recall'].update({expl: [np.mean(recall[expl]), np.std(recall[expl]), sp.stats.ttest_ind(recall[expl], recall[test_against])[1].round(4)]})
  print('')
  print('F1:')
  for expl in explainer_names:
    print(expl, np.mean(f1[expl]), '+-', np.std(f1[expl]), 'pvalue', sp.stats.ttest_ind(f1[expl], f1[test_against])[1].round(4))
    results['F1'].update({expl: [np.mean(f1[expl]), np.std(f1[expl]), sp.stats.ttest_ind(f1[expl], f1[test_against])[1].round(4)]})

  #Calculate initial explanation accuracy
  for expl in explainer_names:
    acc = accuracy[expl]
    acc = sum(acc)/len(acc)
    accuracy[expl] = acc

  return results, round((datetime.datetime.now()-startTime).total_seconds()/60,3), diff, accuracy

if __name__ == "__main__":
    main()
