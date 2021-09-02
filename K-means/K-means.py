import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data_file = os.getcwd() + "\\iris.csv"
file = np.array(pd.read_csv(data_file))[:, :-1]

from sklearn.utils import shuffle
data = shuffle(file)
print(data)


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

from sklearn.cluster import KMeans

K = 3
kmeans = KMeans(K, random_state=0).fit(data)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(data)
print(pred_label)
kmeans_display(data, pred_label)