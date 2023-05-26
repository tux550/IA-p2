from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from scipy.stats import kendalltau
from scipy.special import kl_div

def evaluate_clustering(model, X):
  # Train & Predict
  labels = model.fit(X)

  # Metrics 
  # TODO
  # Kendall Tau
  # KL div
  pass

def evaluate_clustes(model, X):
  # Train & Predict
  labels = model.fit(X)

  # Metrics 
  # TODO
  # Single linkeage
  # Complete linkage
  # Centrod Method
  # Average linkage
  pass

def evaluate_prediction(model, X, Y):
  # Train & Predict
  labels = model.fit(X)

  # Metrics 
  homo = homogeneity_score(Y.reshape(-1), labels)
  comp = completeness_score(Y.reshape(-1), labels)
  vmsr = v_measure_score(Y.reshape(-1), labels)

  # TODO
  # Matriz de similitud

  # Display
  #  display(X, labels, Y, model_name)

  # Return
  metrics = {
    "homogeneity_score" : homo,
    "completeness_score": comp, 
    "v_measure_score"   : vmsr
  }
  return metrics

# Extra TODO
# Cohesion
# Modularidad
# Silohutte value
# Pureza del cluster
# Entropia
# Tabla de contingencia
# ...
