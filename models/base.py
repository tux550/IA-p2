import abc

class ClusteringMethod(abc.ABC):
  @abc.abstractmethod
  def fit(self,X):
    pass
