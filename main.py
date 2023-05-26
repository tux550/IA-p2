from load import load_imgs, load_iris
from models import DBScan, GMM, KMeans
from misc.plot_dist import plot_dist
from misc.display import display

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
  labels = m.fit(X)
  print(model_name)
  print(labels)
  print(len(labels))
  print(Y.reshape(-1))
  display(X, labels, Y, model_name)