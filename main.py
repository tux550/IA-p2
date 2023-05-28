import numpy as np
from load import load_imgs, generate_transformer
from models import DBScan, GMM, KMeans
from misc.plot_dist import plot_dist
from misc.display import display
from test.test import test_all, test_compare

# Set print options
np.set_printoptions(precision = 5, floatmode="fixed", suppress = True)

# Random Seed
np.random.seed(42)

# Load dataset
print("Loading Dataset ...")
X,Y            = load_imgs(depth=3, img_limit = 20)


# Analyze dataset
#plot_dist(X)

# Transform dataset
fv_transformer = generate_transformer(X, do_minmaxscaler=True, pca_components=2)
X = fv_transformer(X)
print("X Shape:", X.shape)
print("Y Shape:",Y.shape)

# Test Models
print("Testing Models ...")
test_all(X)
test_compare(X,Y)
exit()