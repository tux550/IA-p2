import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table

def get_cmap(n, name='hsv'):
    # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    return plt.cm.get_cmap(name, n)

def table_tagcount(models_tagcount, tagnames):
    table = Table(title=f"Clusters per tag")
    table.add_column("tag", justify="right", style="cyan", no_wrap=True)

    for model_name in models_tagcount:
        table.add_column(f"{model_name}", justify="center", style="green")
    for tag in tagnames:
        row_t = [str(tag),]
        for  model_name in models_tagcount:
            row_t.append( str(models_tagcount[model_name][tag]) )        
        table.add_row(*row_t)
    console = Console(record=True)
    console.print(table)
    console.save_svg("fig/tags_all_models.svg")

def visualize_real(X,Y):
    # PCA to 2D
    pca = PCA(n_components=2)
    Xt = pca.fit_transform(X)
    # Get clusters
    n_tags     = len(np.unique(Y))
    cmap = get_cmap(n_tags)
    plt.clf()
    plt.scatter(Xt.T[0], Xt.T[1], color=[cmap(i) for i in Y])
    plt.title(f"Real Cluster")
    plt.savefig(f"fig/visualize_real.png")
    #plt.show()


def visualize_clusters(X, Y,tagnames, labels, model_name):
    # PCA to 2D
    pca = PCA(n_components=2)
    Xt = pca.fit_transform(X)
    # Get clusters
    clusters     = np.unique(labels)
    clusters_tag = dict() 
    for c in clusters:
        # Get real tag for each cluster
        counts  = np.bincount(Y[labels==c])
        tag     = np.argmax(counts)
        hitrate = counts[tag] / counts.sum()
        # Update clusters tag & hitrate
        clusters_tag[c] = {"tag":tag, "hitrate":hitrate}
    # Display
    n_clusters = len(clusters)
    n_tags     = len(np.unique(Y))
    cmap = get_cmap(n_tags)
    plt.clf()
    plt.scatter(Xt.T[0], Xt.T[1], color=[cmap(clusters_tag[i]["tag"]) for i in labels])
    plt.title(f"Model: {model_name}")
    plt.savefig(f"fig/visualize_clusters_{model_name}.png")
    #plt.show()
    
    emotions_count = {name:0 for name in tagnames}
    for c in clusters_tag:
        name = tagnames[clusters_tag[c]["tag"]]
        emotions_count[name] += 1
    
    return emotions_count

def visualize_models(X, Y, tagnames, model_labels):
    # Visualize real
    visualize_real(X,Y)
    # Visualize clusters
    models_tagcount = dict()
    for model_name in model_labels:
        labels = model_labels[model_name]
        models_tagcount[model_name] = visualize_clusters(X,Y,tagnames, labels, model_name)
    # Cluster count in each
    table_tagcount(models_tagcount, tagnames)
