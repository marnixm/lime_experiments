import evaluate_explanations_func
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pprint

DATASETS = ['multi_polarity_books'] #, 'multi_polarity_dvd', 'multi_polarity_kitchen']
#TODO check l1/l2 regularization
ALGORITHM = ['l1logreg', 'tree']
EXPLAINER = ['shap', 'lime', 'parzen', 'greedy', 'random']
PARAMS_5_2 = {'max_examples': 5, #if None than maximum is used
              'lime': {'num_samples': 200, 'rho': 25},  #nsamples to 15.000
              'shap': {'nsamples': 200, 'K': 10, 'num_features': 'num_features(10)'},  #what K (background data), nsampels?
              'max_iter_logreg': 2000,
              'parzen_num_cv': 5}  #was standard
results =  np.zeros((len(DATASETS), len(ALGORITHM), len(EXPLAINER)))
faith =  np.zeros((len(DATASETS), len(ALGORITHM), len(EXPLAINER)))
path = 'C:/Users/marnix.maas/OneDrive - Accenture/Thesis/Results_5.2/'
resultsfile, calcTimefile, faithfile = 'result5.2.p', 'calcTime5.2.p', 'faith5.2.p'

def run_5_2(save=True):
  if save: open(path + 'parameters.txt', 'w').write(pprint.pformat(PARAMS_5_2)) #write parameters
  totalTime = 0
  calcTimes = np.zeros(len(EXPLAINER))
  for d, dat in enumerate(DATASETS):
    for a, alg in enumerate(ALGORITHM):
      for e, exp in enumerate(EXPLAINER):
        temp = evaluate_explanations_func.main(dat, alg, exp, PARAMS_5_2)
        results[d][a][e] = temp['score']
        faith[d][a][e] = np.nanmean(temp['faithfulness']) #todo box instead of mean
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
  return

def plot_5_2(file, save=False, show=True):
  x = np.arange(len(EXPLAINER))
  width = 0.35
  results = pickle.load(open(path + file, "rb"))
  #bigfig, bigax = plt.subplots(2,3)
  for d, dat in enumerate(DATASETS):
    for a, alg in enumerate(ALGORITHM):
      fig, ax = plt.subplots()
      for e, exp in enumerate(EXPLAINER):
        score = results[d][a][e]*100
        ax.bar(x[e], score, width)

      #set titles and axis
      ax.set_ylim([0, 105])
      ax.set_ylabel('Recall (%)')
      ax.set_title('Evaluate recall'+' '+dat+' '+alg)
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

      if save: plt.savefig(path+dat+' '+alg+'.svg')
      if save: plt.savefig(path+dat+' '+alg+'.png')
      if show: plt.show()
      plt.clf()

  plt.close()
  return

#run_5_2(save=True)
plot_5_2(file=faithfile, save=False, show=True)
