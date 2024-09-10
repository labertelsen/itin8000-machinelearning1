# Lauren Bertelsen
# ACMP 8000 ML 1 Unit 2 Activity

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
digits = load_digits()

# data, target, frame, feature_names, target_names, images, DESCR
# print("digits.keys():\n", digits.keys())

# 1797 samples and 64 features
# print("Shape of digit data:", digits.data.shape)

# 64 features, 8x8 square of pixels
# print(digits.feature_names)

# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# print(digits.target_names)

# print(digits.DESCR)

# split digits into train and test groups (90-10)
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=0, test_size = 0.1)

print("-----K-Nearest Neighbors-----")

# set k = 1
n_neighbors = 1
print("k = 1")

# build KNN classifier model
# https://learning.oreilly.com/library/view/introduction-to-machine/9781449369880/ch02.html#relation-of-model-complexity-to-dataset-size
clf = KNeighborsClassifier(n_neighbors=n_neighbors)

# train with training data
clf.fit(X_train, y_train)

# output accuracies
clf_training_accuracy = clf.score(X_train, y_train)
print("Training accuracy: ", clf_training_accuracy)
clf_test_accuracy = clf.score (X_test, y_test)
print("Testing accuracy: ", clf_test_accuracy)

print("-----Linear Model-----")

# build linear model
lr = linear_model.LogisticRegression(solver='lbfgs', max_iter=400)
lr.fit(X_train, y_train)

# output accuracies
lr_training_accuracy = lr.score(X_train, y_train)
print("Training accuracy: ", lr_training_accuracy)
lr_test_accuracy = lr.score(X_test, y_test)
print("Testing accuracy: ", lr_test_accuracy)

print("-----SVM-----")
# load data
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# remove class 0, leaving classes 1 and 2, only use the first 2 features
iris_X = iris_X[iris_y != 0, :2]
iris_y = iris_y[iris_y != 0]

# randomize order and split into train and test groups
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# build SVM
svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train, iris_y_train)

# output accuracies
svc_training_accuracy = svc.score(iris_X_train, iris_y_train)
print("Training accuracy: ", svc_training_accuracy)
svc_test_accuracy = svc.score(iris_X_test, iris_y_test)
print("Testing accuracy: ", svc_test_accuracy)
