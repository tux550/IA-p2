import numpy as np
from sklearn.neighbors import KDTree
from .base import ClusteringMethod


class DBScan(ClusteringMethod):
  """
  DBScan:

  Hyperparameters
  ---------------
  r       : radius
  min_pts : minimum amount of points to cluster
  dist    : distance
  """

  def __init__(self, r, min_pts, dist="minkowski"):
    # Dist options: minkowski,cityblock,chebyshev
    self.r       = r
    self.min_pts = min_pts
    self.dist    = dist

  def fit(self, X):
    # Create KDTree
    X_tree     = KDTree(X, metric=self.dist)

    # Init labels as undefined
    # Note:
    # label=-1   reservado para "undefined"
    # label=0  reservado para "Noise"
    labels     = -np.ones(len(X))
    next_label = 1


    for p in range(len(X)):
      # Only if undefined
      if labels[p] != -1:
        continue
      # Get neighbours
      x = X[p:p+1]
      neigh = X_tree.query_radius(x, r=self.r)
      neigh = neigh[0]

      # If noise continue
      if len(neigh) < self.min_pts:
        labels[p] = 0
        continue
      # New cluster
      c = next_label
      next_label+=1
      labels[p] = c
      # Expand neighborhood
      S = set(neigh)
      S.discard(p)
      for i in S:
        # Rename noise
        if labels[i] == 0:
          labels[i] = c        
        # If already clustered ignore
        elif labels[i] != -1:
          pass
        else:
          # If new, label and expand
          labels[i] = c
          x = X[i:i+1]
          neigh = X_tree.query_radius(x, r=self.r)[0]
          if len(neigh) >= self.min_pts: # Core-point check
            S = S.union( set(neigh) )
    return labels