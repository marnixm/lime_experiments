import data_trusting
import numpy as np
import pickle
import pprint
import pandas as pd
from explanability_metric import *
import os
import matplotlib.pyplot as plt

DATASETS = [('multi_polarity_books','Books'), ('multi_polarity_dvd','DVDs'), ('multi_polarity_kitchen','Kitchen')]
ALGORITHMS = ['neighbors', 'NN'] #[('logreg', 'LR'), ('tree','Tree'), ('svm','SVM'), ('random_forest' ,'RF')]
EXPLAINERS = [('shap','SHAP'), ('lime','LIME'), ('parzen','Parzen')]
path = os.path.abspath(os.curdir) + '/Results_5.3/'
if True:
  # Use generated data instead of multi polarity
  DATASETS = [('Generated', 'Gen')] * 4

#TODO check and run parameters
PARAMS_5_3 = {'percent_untrustworthy': .25, 'num_rounds': 10,
              'lime': {'num_samples': 1000, 'rho': 25},
              'shap': {'nsamples': 1000, 'n_clusters': 10, 'num_features': 'num_features(10)'},
              'rf': {'n_estimators': 1000}, #n_est: 1000
              'num_features': 10,
              'parzen_num_cv': 5,
              'max_examples': 200, #None
              'test_against': 'shap',
              'Gen_count': 0, #to pick synthetic data parameters
              'Gen1': {'n_inf': 10, 'n_redundant': 0, 'n_features': 50, 'noise': 0.05, 'seed': 1, 'nrows': 2000},
              'Gen2': {'n_inf': 10, 'n_redundant': 20, 'n_features': 50, 'noise': 0.05, 'seed': 1, 'nrows': 2000},
              'Gen3': {'n_inf': 10, 'n_redundant': 0, 'n_features': 50, 'noise': 0.3, 'seed': 1, 'nrows': 2000},
              'Gen4': {'n_inf': 10, 'n_redundant': 20, 'n_features': 50, 'noise': 0.3, 'seed': 1, 'nrows': 2000}}

result_file, time_file = 'result5.3', 'calcTime5.3.p'
F1 = np.zeros((len(DATASETS), len(ALGORITHMS), len(EXPLAINERS)))
Precision = np.zeros((len(DATASETS), len(ALGORITHMS), len(EXPLAINERS)))
Recall = np.zeros((len(DATASETS), len(ALGORITHMS), len(EXPLAINERS)))
Accuracy = np.zeros((len(DATASETS), len(ALGORITHMS), len(EXPLAINERS)))
resultsTotal = [[[ [] for i in range(len(EXPLAINERS))] for j in range(len(ALGORITHMS))] for k in range(len(DATASETS))]
if DATASETS[0] == "Generated": path = path[:-1] + " generated/"

def box(a,b, c, d, title, path):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle(title)
  ax1.boxplot([a, b])
  ax2.boxplot([c, d])
  fig.savefig(path)
  #fig.close()
  fig.clf()
  return

def heat(data):
  sns.heatmap(data, annot=True, fmt="f")
  plt.show()
  #plt.clf()
  #heat(table)

def run_5_3(save=True):
  totalTime = 0
  if save: open(path + 'parameters.txt', 'w').write(pprint.pformat(PARAMS_5_3))  # write parameters
  explainers = list(zip(*EXPLAINERS))[0]
  algorithms = list(zip(*ALGORITHMS))[0]

  for d, dat in enumerate(DATASETS):
    PARAMS_5_3['Gen_count'] += 1
    dat = dat[0]
    for a, alg in enumerate(algorithms):
      print('\n',dat, PARAMS_5_3['Gen_count'] if dat=="Generated" else "", alg)
      temp, time, diff, accuracy = data_trusting.main(dat, alg, PARAMS_5_3)
      Lmean, Ldiscount = zip(*diff['lime'])
      Smean, Sdiscount = zip(*diff['shap'])
      #print('difference to prediction', 'lime',round(np.mean(Lmean),1), 'shap',round(np.mean(Smean),1))
      #print('discount', 'lime', round(np.mean(Ldiscount),1), 'shap', round(np.mean(Sdiscount),1))
      if save: box(Lmean, Smean, Ldiscount, Sdiscount,  "Left: Diff to prediction - Right: Discount",
                   path+'/Boxplots/boxplot_'+dat+str(PARAMS_5_3[Gen_count]) if dat=="generated" else ""+'_'+alg+'.png')
      print('Time:',time)
      totalTime+=time
      for e, exp in enumerate(explainers):
        F1[d][a][e] = temp['F1'][exp][0] #[0]:mean, [1]:std, [2]:p-value
        Precision[d][a][e] = temp['Precision'][exp][0]
        Recall[d][a][e] = temp['Recall'][exp][0]
        Accuracy[d][a][e] = accuracy[exp]
        resultsTotal[d][a][e].append(temp['F1'][exp])  # [0]:mean, [1]:std, [2]:p-value

  print('total time:', totalTime)
  if save:
    PARAMS_5_3['Gen_count'] = 0
    pickle.dump(F1, open(path + result_file + '.p', "wb"))
    pickle.dump(resultsTotal, open(path + result_file + '_total.p', "wb"))
    #pickle.dump(calcTimes, open(path + filename2, "wb"))
    for d, dat in enumerate(DATASETS):
      PARAMS_5_3['Gen_count'] += 1
      dat = dat[1] + (str(PARAMS_5_3['Gen_count']) if dat[1]=="Gen" else "")
      pickle.dump(F1[d], open(path + "Datasets/" + dat + '_F1.p', "wb"))
      pickle.dump(Precision[d], open(path + "Datasets/" + dat + '_precision.p', "wb"))
      pickle.dump(Recall[d], open(path + "Datasets/" + dat + '_recall.p', "wb"))
      pickle.dump(Accuracy[d], open(path + "Datasets/" + dat + '_accuracy.p', "wb"))

def table_5_3(stats = ['F1'], save=False, f2=False):
  PARAMS_5_3['Gen_count'] = 0
  explainNames = list(zip(*EXPLAINERS))[1]
  algNames = list(zip(*ALGORITHMS))[1]
  #resTotal = pickle.load(open(path + filename1 + '_total.p', 'rb'))

  fullPath = path+'result5.3_Tables.txt'
  if save:
    if os.path.exists(fullPath): os.remove(fullPath)
    for dataset in DATASETS:
      PARAMS_5_3['Gen_count'] += 1
      dataset = dataset[1] + (str(PARAMS_5_3['Gen_count']) if dataset[1] == "Gen" else "")
      for stat in stats:
        res = pickle.load(open(path + "Datasets/" + dataset +'_'+stat+'.p', "rb"))
        print(fullPath, '\nDataset:', dataset, '-', stat)
        table = pd.DataFrame(res*100, columns=explainNames, index=algNames).transpose().round(1)
        print(fullPath,table)
    print(fullPath, pprint.pformat(PARAMS_5_3))

  else:
    for dataset in DATASETS:
      PARAMS_5_3['Gen_count'] += 1
      dataset = dataset[1] + (str(PARAMS_5_3['Gen_count']) if dataset[1] == "Gen" else "")
      for stat in stats:
        res = pickle.load(open(path + "Datasets/" + dataset +'_'+stat+'.p', "rb"))
        print('\nDataset:', dataset, '-', stat)
        print(pd.DataFrame(res*100, columns=explainNames, index=algNames).transpose().round(1))

  if f2:
    #measure that includes accuracy
    PARAMS_5_3['Gen_count'] = 0
    for dataset in DATASETS:
      PARAMS_5_3['Gen_count'] += 1
      dataset = dataset[1] + (str(PARAMS_5_3['Gen_count']) if dataset[1] == "Gen" else "")

      f1 = pickle.load(open(path + "Datasets/" + dataset +'_f1.p', "rb"))
      acc = pickle.load(open(path + "Datasets/" + dataset + '_accuracy.p', "rb"))
      f1 = pd.DataFrame(f1, columns=explainNames, index=algNames).transpose()
      acc = pd.DataFrame(acc, columns=explainNames, index=algNames).transpose()
      f1adj = f1 * acc * 100

      print('\nDataset:', dataset, '- f1 adjusted', )
      print(f1adj.round(1))
  return

run_5_3(save=True)
table_5_3(stats=[], save=False, f2=True) #'precision', 'recall', 'accuracy'