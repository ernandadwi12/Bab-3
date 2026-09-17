#Exercise 3.7 Principal Component Analysis Breast Cancer - Modified from Example 3.10
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

#Load Breast Cancer data
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

#Plot original data (using first two features: mean radius & mean texture)
f = plt.figure(1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.title('Original Data')

#Perform PCA
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

#Plot PCA data
g = plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data')

plt.show()
