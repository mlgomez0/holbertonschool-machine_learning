#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

pca_transpose = np.transpose(pca_data)
X = pca_transpose[0]
Y = pca_transpose[1]
Z = pca_transpose[2]
labels_float = labels[:].astype(np.float32)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cm = plt.cm.get_cmap('plasma')
ax.scatter(X, Y, Z, c=labels_float, cmap=cm)
ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")
plt.title("PCA of Iris Dataset")
plt.show()
