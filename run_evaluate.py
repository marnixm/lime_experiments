import evaluate_explanations_function
import evaluate_explanations_function_improved
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pprint
import os

DATASETS = ['multi_polarity_books', 'multi_polarity_dvd', 'multi_polarity_kitchen']
DATA_NAMES = ['Books','Dvds','Kitchen']
ALGORITHM = ['l1logreg', 'tree']
ALG_NAMES = ['Logistic regression','Decision tree']
EXPLAINER = ['shap', 'lime', 'parzen']# , 'greedy', 'random']
PARAMS_5_2 = {'max_examples': None, #if None than maximum is used
              'lime': {'num_samples': 2000, 'rho': 25},  #nsamples to 15.000
              'shap': {'nsamples': 2000, 'K': 10, 'num_features': 'num_features(10)'},  #nsampels
              'max_iter_logreg': 2000,
              'parzen_num_cv': 5}  #was standard
experiment = "improved"
results =  np.zeros((len(DATASETS), len(ALGORITHM), len(EXPLAINER)))
faith =  np.zeros((len(DATASETS), len(ALGORITHM), len(EXPLAINER)))
ndcg =  np.zeros((len(DATASETS), len(ALGORITHM), len(EXPLAINER)))
path = os.path.abspath(os.curdir) + '/Results_5.2/'
resultsfile, calcTimefile, faithfile, ndcgfile = 'result5.2.p', 'calcTime5.2.p', 'faith5.2.p', 'ndcg5.2.p'
if experiment == "improved": path = path[:-1] + "_improved/"

def run_5_2(save=True):
  if save: open(path + 'parameters.txt', 'w').write(pprint.pformat(PARAMS_5_2)) #write parameters
  totalTime = 0
  calcTimes = np.zeros(len(EXPLAINER))
  for d, dat in enumerate(DATASETS):
    for a, alg in enumerate(ALGORITHM):
      for e, exp in enumerate(EXPLAINER):
        if experiment=="original":
          temp = evaluate_explanations_function.main(dat, alg, exp, PARAMS_5_2)
        elif experiment=="improved":
          temp = evaluate_explanations_function_improved.main(dat, alg, exp, PARAMS_5_2)
        else:
          print("wrong experiment name")
          return
        results[d][a][e] = temp['score']
        faith[d][a][e] = np.nanmean(temp['faithfulness']) #todo box instead of mean
        print(temp['ndcg'])
        ndcg[d][a][e] = np.mean(temp['ndcg'])
        totalTime += temp['calcTime']
        calcTimes[e] += temp['calcTime']
  print('\ntotalTime', totalTime)
  print('num examples', PARAMS_5_2['max_examples'])
  print(results)
  print(faith)
  if save:
    pickle.dump(results, open(path + resultsfile, "wb"))
    pickle.dump(calcTimes, open(path + calcTimefile, "wb"))
    pickle.dump(faith, open(path + faithfile, "wb"))
    pickle.dump(ndcg, open(path + ndcgfile, "wb"))
  return

def plot_5_2(file, save=False, show=True):
  x = np.arange(len(EXPLAINER))
  width = 0.35
  results = pickle.load(open(path + file, "rb"))
  if file.find('faith')!=-1:
    measure = 'faithfulness'
  elif file.find('result')!= -1:
    measure = 'recall'
  elif file.find('ndcg')!=-1:
    measure = 'ndcg'
  else:
    print('wrong file')
    return
  ncol = results.shape[0]
  nrow = results.shape[1]
  nexp = results.shape[2]
  bigfig, bigax = plt.subplots(nrow, ncol, sharey=True, figsize=(12,8))
  bigax[0,1].set_title('Evaluate - '+ measure + ' %', fontsize=20)

  for d, dat in enumerate(DATA_NAMES):
    for a, alg in enumerate(ALG_NAMES):
      ax = bigax[a,d]

      # set x and y labels
      if d==0: ax.set_ylabel(alg, fontsize=15)
        #todo add small label: recall% or faith%
        #plt.rc('text', usetex=True)
        #ax.set_ylabel(r'{\fontsize{30pt}{3em}\selectfont{}{'+alg+'\r}{\fontsize{18pt}{3em}\selectfont{}'+measure+' %}')
      if a==1: ax.set_xlabel(dat, fontsize=15)

      for e, exp in enumerate(EXPLAINER):
        score = results[d][a][e]*100
        ax.bar(x[e], score, width)

      #set axis
      ax.set_ylim([-105 if np.any(results<0) else 0, 105])
      ax.set_xticks(x)
      ax.set_xticklabels(EXPLAINER)
      ax.set_yticklabels([])

      #set data labels
      for a, y in zip(x, results[d][a]):
        y=round(y*100,1)
        ax.annotate("{:.1f}".format(y),  # this is the text
                     (a, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

  plt.tight_layout()
  if save: plt.savefig(path+measure+' 52'+'.png') #.svg
  if show: plt.show(block=True)
  plt.close()
  return

run_5_2(save=True)
plot_5_2(file=resultsfile, save=True, show=True)
plot_5_2(file=faithfile, save=True, show=True)
plot_5_2(file=ndcgfile, save=True, show=True)