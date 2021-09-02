# Implementation of matplotlib function
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os

path = os.getcwd() + "\\yalefaces\\"


def rename():
    folder = path
    count = 1
    # count increase by 1 in each iteration
    # iterate all files from a directory
    for file_name in os.listdir(folder):
        # Construct old file name
        source = folder + '/' + file_name

        # Adding the count to the new file name and extension
        destination = source + ".pgm"

        # Renaming the file
        os.rename(source, destination)
        count += 1
    # print('All Files Renamed')
    #
    # print('New Names are')
    # # verify the result
    # res = os.listdir(folder)
    # print(res)


# rename()


# read data
states = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy',
          'surprised', 'wink']
prefix = 'subject'
surfix = '.pgm'
h = 243  # height
w = 320  # width
D = h * w
N = len(states) * 15
X = np.zeros((D, N))
cnt = 0
for person_id in range(1, 16):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        X[:, cnt] = matplotlib.pyplot.imread(fn).reshape(D)
        cnt += 1

# Doing PCA, each row is a datapoint
from sklearn.decomposition import PCA

pca = PCA(n_components=100)  # K = 100
pca.fit(X.T)

# projection matrix
U = pca.components_.T

import matplotlib.pyplot as plt


def visualize():
    for i in range(U.shape[1]):
        plt.axis('off')
        f1 = plt.imshow(U[:, i].reshape(h, w), interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = 'eigenface' + str(i).zfill(2) + '.png'
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        # plt.show()

    # See reconstruction of first 6 persons
    for person_id in range(1, 7):
        for state in ['centerlight']:
            fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
            im = matplotlib.pyplot.imread(fn)
            plt.axis('off')
            f1 = plt.imshow(im, interpolation='nearest')
            f1.axes.get_xaxis().set_visible(False)
            f1.axes.get_yaxis().set_visible(False)
            plt.gray()
            fn = 'ori' + str(person_id).zfill(2) + '.png'
            plt.savefig(fn, bbox_inches='tight', pad_inches=0)
            plt.show()
            # reshape and subtract mean
            x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)
            # encode
            z = U.T.dot(x)
            # decode
            x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)

            # reshape to orginal dim
            im_tilde = x_tilde.reshape(h, w)
            plt.axis('off')
            
            f1 = plt.imshow(im_tilde, interpolation='nearest')
            f1.axes.get_xaxis().set_visible(False)
            f1.axes.get_yaxis().set_visible(False)
            plt.gray()
            fn = 'res' + str(person_id).zfill(2) + '.png'
            plt.savefig(fn, bbox_inches='tight', pad_inches=0)
            plt.show()


# visualize()
