import numpy as np
from .utils import test_parameter, compare_models
from models import KMeans, GMM, DBScan

def test_kmeans(X, seed=42):
    # Test NClusters
    init_dict  = {"epochs":100}
    param_name = "n_clusters"
    param_ls   = [2,4,8,16,32,64,128] #[2,4,8,16,32,64,128] #[2,3,4,5,6,7,8,9,10]
    model = KMeans
    test_parameter(X, model, param_name, param_ls, init_dict,seed=seed)

def test_gmm(X, seed=42):
    # Test NClusters
    init_dict  = {"epochs":100}
    param_name = "n_clusters"
    param_ls   = [2,4,8,16,32,64,128] #[2,3,4,5,6,7,8,9,10]
    model = GMM
    test_parameter(X, model, param_name, param_ls, init_dict,seed=seed)

def test_dbscan(X, seed=42):
    # Test R
    init_dict  = {"r":1, "min_pts":3, "dist":"minkowski"}
    param_name = "r"
    param_ls   = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5]
    model = DBScan
    test_parameter(X, model, param_name, param_ls, init_dict,seed=seed)
    # Test min_pts
    init_dict  = {"r":1, "min_pts":3, "dist":"minkowski"}
    param_name = "min_pts"
    param_ls   = [3,4,5,6,7,8,9,10,11,12,13,14,15]
    model = DBScan
    test_parameter(X, model, param_name, param_ls, init_dict,seed=seed)

def test_all(X, seed=42):
    test_gmm(X, seed=seed)
    test_kmeans(X, seed=seed)
    test_dbscan(X, seed=seed)

def test_compare(X,Y, seed=42):
    models = [
        DBScan(1,3,"minkowski"),
        KMeans(100,32),
        GMM(100,32),
    ]
    return compare_models(X,Y, models, seed=seed)
