import matplotlib.pyplot as plt

def display(X, labels, y, model_name):
  plt.clf()
  x_axis = X.T[0]
  y_axis = X.T[1]

  colors = ["black","blue","red","green","orange","purple","pink","cyan","magenta","brown","gray","yellow","beige",]

  fig, (pred_ax, real_ax) = plt.subplots(1, 2)
  fig.suptitle(f'Clustering - Prediction vs Real ({model_name})')
  pred_ax.scatter(x_axis, y_axis, color=[colors[int(i)] for i in labels])
  real_ax.scatter(x_axis, y_axis, color=[colors[int(i+1)] for i in y])
  plt.show()
  