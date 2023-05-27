
from .utils import img2fv
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def generate_transformer(X, do_minmaxscaler=False, pca_components=None):
    # Scale if True
    if do_minmaxscaler:
        minmax_scaler  = MinMaxScaler()
        minmax_scaler.fit(X)
        X = minmax_scaler.transform(X)
    # PCA if True
    if pca_components:
        pca_scaler = PCA(n_components=pca_components)
        pca_scaler.fit(X)
    # Create funciton
    def fv_tranformer(input_X):
        if do_minmaxscaler:
            input_X = minmax_scaler.transform(input_X)
        if pca_components:
            input_X = pca_scaler.transform(input_X)
        return input_X
    # Return function
    return fv_tranformer