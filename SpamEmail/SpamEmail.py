import pandas as pd
import numpy as np
import os


def get_data():
    data_file = os.getcwd() + "\\spambase.csv"
    file = np.array(pd.read_csv(data_file, header=None))
    return file


def save_output(filename, x):
    file = open(filename, mode='w+')
    file.write(str(x))
    file.close()
    return file


def sigmoid(scores):
    model = 1/(1+np.exp(-scores))
    return model

# scores = np.dot(features, weights)


def Stochastic_Descent(X, Y, learning_rate, param):
    if param != 0 and param != 1:
        print("param should be 1 or 0")
        return
    if param == 1:  # co tham so tu do
        new_col = np.array(np.array([1 for i in range(len(X))]))
        X = np.insert(X, 0, new_col, axis=1)

    N, d = X.shape
    theta = np.zeros((d, 1))

    theta_new = np.copy(theta)
    while True:
        k = np.random.permutation(N)
        for i in k:
            for j in range(0, d-1):
                theta_new[j] = theta_new[j] - learning_rate*(sigmoid(np.dot(X[i], theta_new)) - Y[i]) * X[i][j]
        if np.linalg.norm(theta_new - theta) / len(theta) <= 1e-3:
            theta = theta_new
            break
    return theta


file = get_data()
X1 = file[:, :-1]
X2 = file[:, :-3]
Y = file[:, -1:]

save_output("57 features with free-param", Stochastic_Descent(X1, Y, 1e-6, 1))
save_output("55 features with free-param", Stochastic_Descent(X2, Y, 1e-6, 1))
save_output("57 features without free-param", Stochastic_Descent(X1, Y, 1e-6, 0))
save_output("55 features without free-param", Stochastic_Descent(X2, Y, 1e-6, 0))