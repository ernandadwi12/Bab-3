#Exercise 3.13 - Modified Example 3.20: K-means Clustering using make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data points using make_blobs instead of a manual array
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                        cluster_std=1.0, random_state=0)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.predict([[0, 0]]))

# Visualize the clusters and centroids
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', label="Data Points")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label="Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-means Clustering on make_blobs Data")
plt.legend()
plt.grid(True)
plt.savefig("kmeans_plot.png", dpi=150)
plt.show(block=True)