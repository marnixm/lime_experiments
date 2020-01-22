import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# def print(path, *args):
#   text = ' '.join([str(arg) for arg in args])
#   print(text)
#   with open(path, 'a') as log:
#     log.write(text + '\n')

def faithfulness(explanation, skmodel, instance):
  """
  explanability metric: faithfulness
  """
  if len(explanation)<=2:
    #Faithfulness not defined
    return np.nan
  model = []
  explain = []

  for idx, value in explanation:
    # discount removing feature
    ins = instance.copy()
    ins[0,idx] = 0 #background
    model.append(skmodel.predict_proba(ins)[0][0])
    explain.append(value)

  if len(set(explain))<=1 or len(set(model))<=1:
    # Correlation between list and point does not exist
    """explanation and model have no intersection
    inter = set([x[0] for x in explanation]).intersection(set(instance.nonzero()[1]))
    print(inter)"""
    return np.nan

  c = np.corrcoef(model, explain)
  """plt.scatter(model, explain)
  plt.show()"""
  return -c[0, 1]
