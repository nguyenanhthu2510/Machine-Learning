import pandas as pd
import numpy as np
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt

data_file = os.getcwd() + "\\buddymove_holidayiq.csv"
file = pd.read_csv(data_file)
X = np.array(file.transpose())[1:, :]
X = X.astype(float)
N = X.shape[1]
d = X.shape[0]

'''step 1'''
X_average = np.average(X, axis=1).reshape((d, 1))  # vector kỳ vọng

'''step 2'''
Xhat = (X - X_average)  # vector kỳ vọng

'''step 3'''
Sigma = np.dot(Xhat, Xhat.T) / N  # ma trận hiệp phương sai

U, S, V = LA.svd(Sigma)

'''step 5'''
K1 = 5
K2 = 6
U_k1 = Sigma[:, :K1]
U_k2 = Sigma[:, :K2]

'''step 6'''
Z1 = np.dot(U_k1.T, Xhat)  # la vector chứa K thanh phan chinh
Z2 = np.dot(U_k2.T, Xhat)

approx_X = np.dot(U_k1, Z1) + Xhat  # vector xap xi vector ban dau
# approx_X = np.dot(U_k2, Z2) + Xhat  # vector xap xi vector ban dau


def visualize(X, str, K = ""):
    title = str + K
    plt.plot(X[0, :N], X[1, :N], 'o', markersize=7)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    plt.title(title)
    plt.savefig('Pic{}.png'.format(K), bbox_inches='tight', dpi=600)
    plt.show()


visualize(X, "Điểm dữ liệu trong dữ liệu gốc")
visualize(Z1, "Điểm dữ liệu kết quả trong không gian mới với K1 = ", str(K1))
visualize(Z2, "Điểm dữ liệu kết quả trong không gian mới với K2 = ", str(K2))
