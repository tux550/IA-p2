
import os
import numpy as np
from .utils import img2fv, paths_dict
from config import DATASET_PATH
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import datasets

def load_imgs(img_limit = None, do_minmaxscaler=False, pca_components=None):
    # Init X, Y dataset
    X = []
    Y = []
    # Current Label
    curr_label = 1
    # Load imgpaths
    emotions_dict = paths_dict(img_limit=img_limit)    
    # Load dataset
    for emotion in emotions_dict:
        imgpath_ls = emotions_dict[emotion]
        for img_name in imgpath_ls:
            img_path = DATASET_PATH + "/"+emotion+"/" + img_name
            fv = img2fv(img_path)
            X.append(fv)
            Y.append(curr_label)
        curr_label += 1
    # Scale if True
    if do_minmaxscaler:
        scaler  = MinMaxScaler()
        X = scaler.fit_transform(X)
    # PCA if True
    if pca_components:
        pca = PCA(n_components=20)
        X = pca.fit_transform(X)
    # Dataset to np array
    X = np.array(X)
    Y = np.array(Y).reshape(-1,1)
    # Return
    return X,Y

def load_iris():
    iris = datasets.load_iris()
    X = iris.data[:,2:4]
    Y = iris.target.reshape(-1,1)
    return X, Y