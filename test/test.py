import numpy as np
from .utils import test_parameter, compare_models
from models import KMeans, GMM, DBScan

def test_kmeans(X):
    # Seed
    np.random.seed(42)
    # Test NClusters
    init_dict  = {"epochs":100}
    param_name = "n_clusters"
    param_ls   = [2,3,4,5,6,7,8,9,10]
    model = KMeans
    test_parameter(X, model, param_name, param_ls, init_dict)

def test_gmm(X):
    # Seed
    np.random.seed(42)
    # Test NClusters
    init_dict  = {"epochs":100}
    param_name = "n_clusters"
    param_ls   = [2,3,4,5,6,7,8,9,10]
    model = GMM
    test_parameter(X, model, param_name, param_ls, init_dict)

def test_dbscan(X):
    # Seed
    np.random.seed(42)
    # Test R
    init_dict  = {"r":1, "min_pts":3, "dist":"minkowski"}
    param_name = "r"
    param_ls   = [1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]
    model = DBScan
    test_parameter(X, model, param_name, param_ls, init_dict)

def test_all(X):
    test_kmeans(X)
    test_gmm(X)
    test_dbscan(X)

def test_compare(X,Y):
    models = [
        KMeans(100,2),
        GMM(100,9),
        DBScan(2,3,"minkowski"),
    ]
    compare_models(X,Y, models)
