import os
import sys
import numpy as np
import matplotlib.pyplot as plt
###for saving results e.d.

def printLog(path, *args):
  text = ' '.join([str(arg) for arg in args])
  print(text)
  with open(path, 'a') as log:
    log.write(text + '\n')


def faithfulness(explanation, skmodel, instance):
  #TODO check method
  """
  explanability metric: faithfulness
  """
  if len(explanation)<=1:
    #what to do with too small explanation?
    return np.nan
  model = []
  explain = []
  #initial probs
  model.append(skmodel.predict_proba(instance)[0][0])
  exp_prob = sum([x[1] for x in explanation])
  explain.append(exp_prob)

  for idx, value in explanation:
    # discount removing feature
    instance[0,idx] = 0 #background
    model.append(skmodel.predict_proba(instance)[0][0])
    exp_prob= exp_prob - value
    explain.append(exp_prob)

  #if len(model)>5:
  #  plt.scatter(model, explain)
  #  plt.show()
  #  print(explanation)
  #return correlation
  c = np.corrcoef(model, explain)
  return abs(c[0, 1])
