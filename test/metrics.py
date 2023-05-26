from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

def get_metrics(model, X, Y):
  # Train & Predict
  labels = model.fit(X)
  # Metrics 
  homo = homogeneity_score(Y.reshape(-1), labels)
  comp = completeness_score(Y.reshape(-1), labels)
  vmsr = v_measure_score(Y.reshape(-1), labels)
  # Display
  #  display(X, labels, Y, model_name)
  # Return
  metrics = {
    "homogeneity_score" : homo,
    "completeness_score": comp, 
    "v_measure_score"   : vmsr
  }
  return metrics