import numpy as np
from load import load_imgs, generate_transformer
from models import DBScan, GMM, KMeans
from misc.plot_dist import plot_dist
from misc.display import display
from test.test import test_all, test_compare

# Config
depth = 1#1, 3
pca_components=3#20, 2

# Set print options
np.set_printoptions(precision = 5, floatmode="fixed", suppress = True)

# Random Seed
np.random.seed(42)

# Load dataset
print("Loading Dataset ...")
X,Y            = load_imgs(depth=depth)

# Transform dataset
fv_transformer = generate_transformer(X, do_minmaxscaler=True, pca_components=pca_components)
X = fv_transformer(X)
print("X Shape:", X.shape)
print("Y Shape:",Y.shape)

# Test Models
print("Testing Models ...")
#test_all(X,seed=42)
test_compare(X,Y,seed=42)
exit()