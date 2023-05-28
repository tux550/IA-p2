import numpy as np
from load import load_imgs, load_iris, generate_transformer
from models import DBScan, GMM, KMeans
from misc.plot_dist import plot_dist
from misc.display import display
from test.metrics import evaluate_clustering, evaluate_prediction

from scipy.spatial.distance import minkowski, cityblock, chebyshev


############################################################
#
#                      Random Seed
# 
############################################################
np.random.seed(42)


############################################################
#
#                      Load Dataset
# 
############################################################
print("Loading Dataset ...")
X,Y = load_imgs(depth=3, img_limit = 20) #DEBUG with iris: X, Y = load_iris()
fv_transformer = generate_transformer(X, do_minmaxscaler=True, pca_components=2)
X = fv_transformer(X)
print(X.shape)
print(Y.shape)


############################################################
#
#                      Config Models
# 
############################################################
# Real Models
models = {
    #"GMM": GMM(100, 7),
    "KMeans": KMeans(100, 35, distance_method=cityblock),
    #"DBScan": DBScan(r=1, min_pts=8, dist="minkowski"),
}
"""
# Iris Models
models = {
    "GMM": GMM(100, 3),
    "KMeans": KMeans(10, 3),
    "DBScan": DBScan(r=1, min_pts=8, dist="minkowski"),
}
"""

############################################################
#
#                      Test Models
# 
############################################################
print("Testing Models ...")
for model_name in models:
  # Model
  m      = models[model_name]
  print(model_name)
  # Metrics 
  metrics = evaluate_prediction(m,X,Y)
  print("Metrics", metrics)