import numpy as np
from scipy.spatial.distance import minkowski
from .base import ClusteringMethod

class KMeans(ClusteringMethod):
  

  def __init__(self, epochs, n_clusters, distance_method=minkowski):
    self.epochs     = epochs
    self.n_clusters = n_clusters
    self.distance_method = distance_method
    # Extra
    self.reuse = False

  def fit(self, X):
    self.init_centroids(X)
    grupos = None
    for e in range(self.epochs):
        if (e%20 == 0):
          print(f">> epoch: {e}")
        grupos = self.get_groups(X)
        new_c = self.new_centroids(X, grupos)
        #diff = np.array(self.centroids) - np.array(new_c)
        self.centroids = new_c
    return grupos

  def get_groups(self, X):
    groups = []
    for x in X:
        dist = np.array( [self.distance_method(c, x) for c in self.centroids])
        g    = np.argmin( dist )
        groups.append(g)
    groups = np.array(groups)
    return groups


  def new_centroids(self, X, grupos):
    new_c = []
    for c in range(self.n_clusters):
        elements = X[grupos == c]
        if len(elements) > 0:
          new_c.append(np.mean(elements, axis = 0))
        else:
          if not self.reuse:
            print("REUSE")
            self.reuse=True
          new_c.append(self.centroids[c])
    new_c = np.array(new_c)

    return new_c

  def init_centroids(self, X):
    # Select centoids
    self.centroids = X[np.random.choice(len(X), self.n_clusters, replace=False)]
   
