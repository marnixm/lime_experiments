import os
import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *

def faithfulness(explanation, skmodel, instance, perturb):
  """
  explanability metric: faithfulness
  Oneliner. Measure if feature attribution has same effect as model when feature is perturbed
  """
  if len(explanation)<=2:
    #Faithfulness not defined
    return np.nan
  model = []
  explain = []

  for idx, value in explanation:
    # discount removing feature
    ins = instance.copy()
    if perturb is not None:
      #generated data
      ins[idx] = perturb[idx]
      ins = [ins]
    else:
      ins[0,idx] = 0
    model.append(skmodel.predict_proba(ins)[0][0])
    explain.append(value)

  if len(set(explain))<=1 or len(set(model))<=1:
    # Correlation between list and point does not exist,
    # thus faithfulness is not defined
    return np.nan

  c = np.corrcoef(model, explain)
  return -c[0, 1]

def ndcg_score(true_features, exp_features):
  """
  Explanability metric: ndcg
  Oneliner. Compares ranking of features in model to explainer ranking, ranked by feature importance/attribution
  """
  # cut-off the feature vectors at minimum lenght
  # accounted for by multiplying ndcg with recall
  minlength = min(len(true_features), len(exp_features))
  true_features = true_features[:minlength]
  exp_features = exp_features[:minlength]

  if len(true_features) > 1:
    score = sklearn.metrics.ndcg_score([true_features], [exp_features])
  else:
    # length of 1, thus binary score
    score = int(true_features == exp_features)

  return score