import data_trusting
import numpy as np
import pickle
import pprint
import pandas as pd
from utilities import printLog
import os

DATASETS = [('multi_polarity_books','Books')]# , ('multi_polarity_dvd','DVDs')] #, ('multi_polarity_kitchen','Kitchen')]
ALGORITHMS = [('logreg', 'LR') , ('tree','Tree'), ('svm','SVM')]#, ('random_forest' ,'RF')]
EXPLAINERS = [('shap','SHAP'), ('lime','LIME'), ('parzen','Parzen'), ('greedy','Greedy'), ('random','Random')]

#TODO check and run parameters
PARAMS_5_3 = {'percent_untrustworthy': .25, 'num_rounds': 100,
              'lime': {'num_samples': 500, 'rho': 25},  #nsamples to 15.000,
              'shap': {'nsamples': 500, 'K': 10, 'num_features': 'num_features(10)'},  #what K (background data), nsampels?
              'rf': {'n_estimators': 20}, #n_est: 1000
              'num_features': 10,
              'parzen_num_cv': 5,
              'max_examples': None, #None
              'test_against': 'lime'}
path = 'C:/Users/marnix.maas/OneDrive - Accenture/Thesis/Results_5.3/'
result_file, time_file = 'result5.3', 'calcTime5.3.p'
F1 = np.zeros((len(DATASETS), len(ALGORITHMS), len(EXPLAINERS)))
Precision = np.zeros((len(DATASETS), len(ALGORITHMS), len(EXPLAINERS)))
Recall = np.zeros((len(DATASETS), len(ALGORITHMS), len(EXPLAINERS)))
resultsTotal = [[[ [] for i in range(len(EXPLAINERS))] for j in range(len(ALGORITHMS))] for k in range(len(DATASETS))]

#TODO warning at lime sparse matrix
def run_5_3(save=True):

  totalTime = 0
  if save: open(path + 'parameters.txt', 'w').write(pprint.pformat(PARAMS_5_3))  # write parameters
  explainers = list(zip(*EXPLAINERS))[0]
  algorithms = list(zip(*ALGORITHMS))[0]

  for d, dat in enumerate(DATASETS):
    dat = dat[0]
    for a, alg in enumerate(algorithms):
      print('\n',dat, alg)
      temp, time = data_trusting.main(dat, alg, PARAMS_5_3)
      print('Time:',time)
      totalTime+=time
      for e, exp in enumerate(explainers):
        F1[d][a][e] = temp['F1'][exp][0] #[0]:mean, [1]:std, [2]:p-value
        Precision[d][a][e] = temp['Precision'][exp][0]
        Recall[d][a][e] = temp['Recall'][exp][0]
        resultsTotal[d][a][e].append(temp['F1'][exp])  # [0]:mean, [1]:std, [2]:p-value

    print('total time:', totalTime)
    if save:
      pickle.dump(F1, open(path + result_file + '.p', "wb"))
      pickle.dump(resultsTotal, open(path + result_file + '_total.p', "wb"))
      #pickle.dump(calcTimes, open(path + filename2, "wb"))
      for d, dat in enumerate(DATASETS):
        pickle.dump(F1[d], open(path + "Datasets/" + dat[1] + '_F1.p', "wb"))
        pickle.dump(Precision[d], open(path + "Datasets/" + dat[1] + '_precision.p', "wb"))
        pickle.dump(Recall[d], open(path + "Datasets/" + dat[1] + '_recall.p', "wb"))

def table_5_3(stat = 'F1', save=False):
  explainNames = list(zip(*EXPLAINERS))[1]
  algNames = list(zip(*ALGORITHMS))[1]
  #resTotal = pickle.load(open(path + filename1 + '_total.p', 'rb'))

  fullPath = path+'result5.3_Tables.txt'
  if save:
    for dataset in DATASETS:
      dataset = dataset[1]
      res = pickle.load(open(path + "Datasets/" + dataset +'_'+stat+'.p', "rb"))
      printLog(fullPath, '\nDataset:', dataset)
      table = pd.DataFrame(res*100, columns=explainNames, index=algNames).transpose().round(1)
      printLog(fullPath,table)
    if os.path.exists(fullPath): os.remove(fullPath)
    printLog(fullPath, pprint.pformat(PARAMS_5_3))

  else:
    for dataset in DATASETS:
      dataset = dataset[1]
      res = pickle.load(open(path + "Datasets/" + dataset +'_'+stat+'.p', "rb"))
      print('\nDataset:', dataset)
      print(pd.DataFrame(res*100, columns=explainNames, index=algNames).transpose().round(1))

  return

run_5_3(save=True)
table_5_3(stat='precision', save=False)
table_5_3(stat='recall', save=False)

