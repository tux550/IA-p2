import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
from .base import ClusteringMethod

class GMM(ClusteringMethod):
  def __init__(self, epochs, n_clusters):
    self.epochs     = epochs
    self.n_clusters = n_clusters

  def fit(self, X):
    self.init_gaussian(X)
    labels = self.EM(X)
    return labels

  def init_gaussian(self, X):
    # Get u from KMeans
    kmeans = KMeans(n_clusters=self.n_clusters,n_init="auto").fit(X)
    labels = kmeans.labels_
    u      = kmeans.cluster_centers_
    # Get sigma from covariance KMeans
    sigma = []
    for i in range(self.n_clusters):
      sigma.append(np.cov(X[labels==i].T, bias=False))
    sigma = np.array(sigma)
    # Get pi from KMeans
    pi = np.array( [ (labels==i).sum() for i in range(self.n_clusters)] ) / len(labels)
    # Init u, sigma, pi
    self.u     = u
    self.sigma = sigma
    self.pi    = pi


  def EM(self, X):
    n = X.shape[0]
    k = len(self.pi)

    for e in range(self.epochs):
      # E-step:
      ## Calc Likelihood
      LL = np.zeros([n,k]) # (n*k)
      for i in range(k):
        #print("ui", self.u[i])
        #print("sigmai", self.sigma[i])
        #try:
        mnorm = stats.multivariate_normal(self.u[i], self.sigma[i],  allow_singular=True) # TEST
        #except:
        #  print("u", np.isnan(self.u[i]).any())
        #  print("sigma", np.isnan(self.sigma[i]).any())
        #  exit()
        for j in range(n):
          LL[j][i] = mnorm.pdf(X[j])
      # Multiply by p
      for i in range(k):  
        LL[:,i] = self.pi[i] * LL[:,i]
      # Divide by Sum
      y = LL/LL.sum(axis=1)[:,None] # (k,n)

      # M-step: Update the model parameters
      N  = y.sum(axis=0)
      self.pi = N/n
      for j in range(k):
        y_col    = y[:, j].reshape(-1, 1)
        self.u[j]     = np.sum( y_col * X, axis=0) / N[j]
        diff     = X - self.u[j]
        self.sigma[j] = np.dot((y_col * (diff)).T, diff) / N[j]
    # return labels
    return np.argmax(y, axis=1)
