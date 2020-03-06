import evaluate_explanations_function
import evaluate_explanations_function_improved
import evaluate_explanations_function_gen
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pprint
import os

DATASETS = ['multi_polarity_books', 'multi_polarity_dvd', 'multi_polarity_kitchen']
DATA_NAMES = ['Books','DVDs','Kitchen']
if False:
  # Use generated data instead of multi polarity
  DATASETS = ['Generated'] * 4
  DATA_NAMES = ["Generated " + str(i + 1) for i in range(4)]
ALGORITHM = ['tree', 'l1logreg']
ALG_NAMES = ['Logistic regression','Decision tree']
EXPLAINER = ['Shap', 'Lime', 'Parzen']
# collection of parameters used by the experiment and explainers
PARAMS_5_2 = {'max_examples': None, #if None than maximum is used
              'lime': {'num_samples': 15000, 'rho': 25},  #nsamples to 15.000
              'shap': {'nsamples': 15000, 'n_clusters': 10, 'num_features': 'num_features(10)'},
              'max_iter_logreg': 2000,
              'parzen_num_cv': 5,
              'Gen_count': 0, #to pick synthetic data parameters
              'Gen1': {'n_inf': 10, 'n_redundant': 0, 'n_features': 50, 'noise': 0.05, 'seed': 1, 'nrows': 2000},
              'Gen2': {'n_inf': 10, 'n_redundant': 15, 'n_features': 50, 'noise': 0.05, 'seed': 1, 'nrows': 2000},
              'Gen3': {'n_inf': 10, 'n_redundant': 0, 'n_features': 50, 'noise': 0.3, 'seed': 1, 'nrows': 2000},
              'Gen4': {'n_inf': 10, 'n_redundant': 15, 'n_features': 50, 'noise': 0.3, 'seed': 1, 'nrows': 2000}}

experiment = "improved"
results = [[[ [] for i in range(len(EXPLAINER))] for j in range(len(ALGORITHM))] for k in range(len(DATASETS))]
faith = [[[ [] for i in range(len(EXPLAINER))] for j in range(len(ALGORITHM))] for k in range(len(DATASETS))]
ndcg = [[[ [] for i in range(len(EXPLAINER))] for j in range(len(ALGORITHM))] for k in range(len(DATASETS))]
path = os.path.abspath(os.curdir) + '/Results_5.2/'
resultsfile, calcTimefile, faithfile, ndcgfile = 'result5.2.p', 'calcTime5.2.p', 'faith5.2.p', 'ndcg5.2.p'
if experiment == "original": path = path + "original/"
if experiment == "improved": path = path + "improved/"
if DATASETS[0] == "Generated": path = path[:-1] + " generated/"


def run_5_2(save=True):
  if save: open(path + 'parameters.txt', 'w').write(pprint.pformat(PARAMS_5_2)) #write parameters
  totalTime = 0
  calcTimes = np.zeros(len(EXPLAINER))
  for d, dat in enumerate(DATASETS):
    PARAMS_5_2['Gen_count'] += 1
    for a, alg in enumerate(ALGORITHM):
      for e, exp in enumerate(EXPLAINER):
        if experiment=="original":
          temp = evaluate_explanations_function.main(dat, alg, exp, PARAMS_5_2)
        elif experiment=="improved":
          temp = evaluate_explanations_function_gen.main(dat, alg, exp, PARAMS_5_2)
        else:
          print("wrong experiment name")
          return
        results[d][a][e] = temp['score']
        faith[d][a][e] = temp['faithfulness']
        if experiment=="improved": ndcg[d][a][e] = temp['ndcg']
        totalTime += temp['calcTime']
        calcTimes[e] += temp['calcTime']
  print('\ntotalTime', totalTime)
  print('num examples', PARAMS_5_2['max_examples'])
  print(results)
  print(faith)
  if experiment=="improved": print(ndcg)
  if save:
    pickle.dump(results, open(path + resultsfile, "wb"))
    pickle.dump(calcTimes, open(path + calcTimefile, "wb"))
    pickle.dump(faith, open(path + faithfile, "wb"))
    if experiment=="improved": pickle.dump(ndcg, open(path + ndcgfile, "wb"))
  return

def plot_5_2(file, save=False, show=True, plot='bar'):
  x = np.arange(len(EXPLAINER))
  gencount=0
  width = 0.85
  results = np.array(pickle.load(open(path + file, "rb")))
  neg=False
  if file.find('faith')!=-1:
    measure = 'Faithfulness'
  elif file.find('result')!= -1:
    measure = 'Recall'
  elif file.find('ndcg')!=-1:
    measure = 'NDCG'
  else:
    print('wrong file')
    return
  ncol = results.shape[0]
  nrow = results.shape[1]
  nexp = results.shape[2]
  bigfig, bigax = plt.subplots(nrow, ncol, sharey=True, figsize=(10,6))
  bigfig.suptitle(measure + ' (in %)',fontsize=20)

  for d, dat in enumerate(DATA_NAMES):
    gencount+=1
    for a, alg in enumerate(ALG_NAMES):
      ax = bigax[a,d]
      # set x and y labels
      if d==0: ax.set_ylabel(alg, fontsize=15)
      if a==1:
        if not dat.startswith("Generated"):
          ax.set_xlabel(dat, fontsize=15)
        else:
          ax.set_xlabel('Redundancy: ' + str(PARAMS_5_2['Gen'+str(gencount)]['n_redundant']) +
                        '\nNoise: ' + str(PARAMS_5_2['Gen'+str(gencount)]['noise']),fontsize=15, labelpad=10)

      for e, exp in enumerate(EXPLAINER):
        if plot=='bar':
          score = np.nanmean(results[d][a][e])*100
          if score<0: neg = True
          ax.bar(x[e], score, width, color='lightsteelblue', edgecolor='black') #lightsteelblue
        if plot=='box':
          s = [s*100 for s in results[d][a][e] if not np.isnan(s)]
          if np.any(np.array(s)<0): neg=True
          ax.boxplot(s, positions=[x[e]])

      #set axis
      ax.set_ylim([-110 if neg else 0, 125 if neg else 110])
      ax.set_xticks(x)
      ax.set_xticklabels(EXPLAINER, fontsize=14)
      ax.set_yticklabels([])

      #set data labels
      if plot == 'bar':
        res = [np.nanmean(r) for r in results[d][a]]
        for a, y in zip(x, res):
          y=round(y*100,1)
          ax.annotate("{:.1f}".format(y),  # this is the text
                       (a, y),  # this is the point to label
                       textcoords="offset points",  # how to position the text
                       xytext=(0, (2 if y>0 else -14)),  # distance from text to points (x,y)
                       ha='center', # horizontal alignment can be left, right or center
                       fontsize=14)

  plt.tight_layout()
  plt.subplots_adjust(top=0.90)
  if save: plt.savefig(path+measure+' 52'+ (' box' if plot=='box' else '') +'.png') #.svg
  if show: plt.show(block=True)
  plt.close()
  return

run_5_2(save=False)
save=False
show=True
plot='bar' #plot as bar or boxlplot
plot_5_2(file=resultsfile, save=save, show=show, plot=plot)
plot_5_2(file=faithfile, save=save, show=show, plot=plot)
plot_5_2(file=ndcgfile, save=save, show=show, plot=plot)