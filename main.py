import numpy as np
from load import load_imgs, generate_transformer
from models import DBScan, GMM, KMeans
from misc.plot_dist import plot_dist
from misc.display import display
from test.test import test_all, test_compare
from visualize.visualize import visualize_clusters, visualize_models

### Integrantes:
# Salazar Alva, Rodrigo Gabriel (100 %)
# Nicho Galagarza, Jorge Ernesto (100 %)
# Loayza Pretel, Gabriel Luis Dario (100 %)

###
# BEST CONFIG:
# Depth: 1
# PCA: 10
# KMeans(100,32),
# GMM(100,32),
# DBScan(1,3,"minkowski"),

# Config
depth = 1#1
pca_components= 10#10

# Set print options
np.set_printoptions(precision = 5, floatmode="fixed", suppress = True)

# Random Seed
np.random.seed(42)

# Load dataset
print("Loading Dataset ...")
X,Y, tagnames = load_imgs(depth=depth)

# Transform dataset
fv_transformer = generate_transformer(X, do_minmaxscaler=True, pca_components=pca_components)
X = fv_transformer(X)
print("X Shape:", X.shape)
print("Y Shape:",Y.shape)

# Test Models
print("Testing Models ...")
test_all(X,seed=42)
labels_dict = test_compare(X,Y,seed=42)
visualize_models(X, Y, tagnames,labels_dict)
exit()