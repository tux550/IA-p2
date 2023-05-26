import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

def plot_dist(X, distance_function=minkowski):
    plt.clf()
    all_dist = []
    for x1 in X:
        for x2 in X:
            all_dist.append( distance_function(x1,x2) )
    plt.hist(all_dist)
    plt.show() 