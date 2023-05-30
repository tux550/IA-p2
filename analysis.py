import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
import numpy as np
from load import load_imgs, generate_transformer

# Random Seed
np.random.seed(42)

# Load dataset
print("Loading Dataset ...")


def plt_dist(plt_obj, X, distance_function=minkowski):
    all_dist = []
    for x1 in X:
        for x2 in X:
            all_dist.append( distance_function(x1,x2) )
    plt_obj.hist(all_dist)


depth_ls = (0,1,2,3)
components_ls = (None,20,10,5,3,2)
show = True
fig, subplt_ls_ls = plt.subplots(len(depth_ls), len(components_ls))


for depth, subplt_ls in zip(depth_ls, subplt_ls_ls):
    X,Y,_ = load_imgs(depth=depth)
    for components, subplt in zip(components_ls,  subplt_ls):
        print(f"Running d:{depth} c:{components}")
        if components is None:
            plt_dist(subplt, X)
        elif X.shape[1] >= components:
            fv_transformer = generate_transformer(X, do_minmaxscaler=True, pca_components=components)
            t_X = fv_transformer(X)
            plt_dist(subplt, t_X)
        else:
            subplt.text(0.5, 0.5, 'None', ha='center', va='center', size=20)
        #subplt.set_title(f"d:{depth} - c:{components}")
        subplt.set(xlabel=f'c:{components}', ylabel=f'd:{depth}')
        subplt.label_outer()

fig.savefig(f"fig/plt_dist_matrix.png")
plt.show()
    