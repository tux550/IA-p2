import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
from misc.display import display

def evaluate_clustering(model, X):
  # Train & Predict
  labels = model.fit(X)
  #print("labels",labels)
  # Metrics
  if len(np.unique(labels))>1:
    shsc = silhouette_score(X,labels)
    chsc = calinski_harabasz_score(X, labels)
    dbsc = davies_bouldin_score(X, labels)
  else:
    shsc = None
    chsc = None
    dbsc = None
  # Return
  metrics = {
    "Silhouette Coefficient" : shsc,
    "Calinski-Harabasz Index": chsc, 
    "Davies-Bouldin Index"   : dbsc,
  }
  return metrics

def evaluate_prediction(model, X, Y):
  # Train & Predict
  labels = model.fit(X)

  #print("labels",labels)

  # Metrics 
  if len(np.unique(labels))>1:
    homo = homogeneity_score(Y.reshape(-1), labels)
    comp = completeness_score(Y.reshape(-1), labels)
    vmsr = v_measure_score(Y.reshape(-1), labels)
    rans = adjusted_rand_score(Y.reshape(-1), labels)
  else:
    shsc = None
    chsc = None
    dbsc = None

  # Return
  metrics = {
    "Homogenity "        : homo,
    "Completness"        : comp, 
    "V-measure"          : vmsr,
    "Adjusted Rand Index": rans,
  }
  return metrics, labels



# TODO: Extra
# Matriz de similitud
# Display

