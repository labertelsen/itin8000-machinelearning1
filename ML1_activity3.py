# Example 1 -----------------------------------------------------------------------------
# K-means clustering
import numpy as np
np.random.seed(1)
from sklearn import cluster, datasets
X_iris, y_iris = datasets.load_iris(return_X_y=True)
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)
print(k_means.labels_[::10])
print(y_iris[::10])
# Example 2 -----------------------------------------------------------------------------
# Vector quantization

import scipy as sp
try:
    face = sp.face(gray=True)
except AttributeError:
    from scipy import misc
    face = misc.face(gray=True)
X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=5, n_init=1)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

# Example 3 -----------------------------------------------------------------------------
# Connectivity-Constrained Clustering

from skimage.data import coins
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
rescaled_coins = rescale(gaussian_filter(coins(), sigma=2), 0.2, mode='reflect', anti_aliasing=False)
X = np.reshape(rescaled_coins, (-1, 1))

from sklearn.feature_extraction import grid_to_graph
connectivity = grid_to_graph(*rescaled_coins.shape)

n_clusters = 27
from sklearn.cluster import AgglomerativeClustering
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage = 'ward', connectivity=connectivity)
ward.fit(X)
label = np.reshape(ward.labels_, rescaled_coins.shape)

# Example 4 -----------------------------------------------------------------------------
# Feature Agglomeration

digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=32)
agglo.fit(X)
X_reduced = agglo.transform(X)
X_approx = agglo.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)

# Example 5 -----------------------------------------------------------------------------
# PCA

x1 = np.random.normal(size = (100, 1))
x2 = np.random.normal(size = (100, 1))
x3 = x1+x2
X = np.concatenate([x1, x2, x3], axis=1)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:, 2])
_ = ax.set(xlabel = "x", ylabel ="y", zlabel = "z")

from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
print(pca.explained_variance_)
pca.set_params(n_components=2)
X_reduced = pca.fit_transform(X)
X_reduced.shape

# Example 6 -----------------------------------------------------------------------------
# ICA

import numpy as np
from scipy import signal
time = np.linspace(0, 10, 2000)
s1 = np.sin(2*time)
s2 = np.sign(np.sin(2*time))
s3 = signal.sawtooth(2*np.pi*time)
S = np.c_[s1, s2, s3]
S += 0.2*np.random.normal(size=S.shape)
S /= S.std(axis=0)
A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
X = np.dot(S, A.T)

ica = decomposition.FastICA()
S_ = ica.fit_transform(X)
A_ = ica.mixing_.T
np.allclose(X, np.dot(S_, A_) + ica.mean_)
