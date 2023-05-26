from load import load_imgs, load_iris
from models import DBScan, GMM, KMeans
from misc.plot_dist import plot_dist
from misc.display import display
from test.metrics import evaluate_clustering, evaluate_prediction

print("Loading Dataset ...")
#X,Y = load_imgs(img_limit = None, do_minmaxscaler=False, pca_components=None)
X, Y = load_iris()
print(X.shape)
print(Y.shape)


#plot_dist(X)
#exit()

print("Testing Models ...")
models = {
    "GMM": GMM(100, 3),
    "KMeans": KMeans(10, 3),
    "DBScan": DBScan(r=1, min_pts=8, dist="minkowski"),
}

for model_name in models:
  m      = models[model_name]
  # Model
  print(model_name)
  # Metrics 
  metrics = evaluate_prediction(m,X,Y)
  print("Metrics", metrics)
