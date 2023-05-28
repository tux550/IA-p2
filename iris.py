import numpy as np
from load import load_imgs, load_iris, generate_transformer
from models import DBScan, GMM, KMeans
from misc.plot_dist import plot_dist
from misc.display import display
from test.metrics import evaluate_clustering, evaluate_prediction
from test.table import table_results
from scipy.spatial.distance import minkowski, cityblock, chebyshev
from test.test import test_kmeans, test_gmm, test_dbscan, test_all

# Random Seed
np.random.seed(42)

# Load dataset
print("Loading Dataset ...")
X, Y=load_iris()
print("X Shape:", X.shape)
print("Y Shape:",Y.shape)

# Test Models
print("Testing Models ...")
test_all(X)
exit()