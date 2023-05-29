import numpy as np
from .table import table_results
from .metrics import evaluate_clustering, evaluate_prediction

def test_parameter(X, model, param_name, param_ls, init_dict, seed=42):
  # Seed
  np.random.seed(seed)

  result_dict = dict()
  for p in param_ls:
    print(f"> {param_name}: {p}")
    init_dict[param_name] = p
    m              = model(**init_dict)
    result_dict[p] = evaluate_clustering(m,X)
  table_results(model.__name__, param_name, result_dict)

def compare_models(X, Y, models, seed=42):
  # Seed
  np.random.seed(seed)

  result_dict = dict()
  for m in models:
    print(f"> modek: {m.__class__.__name__}")
    result_dict[m.__class__.__name__] = evaluate_prediction(m,X, Y)
  table_results("Compare models", "models", result_dict, format_title=False)