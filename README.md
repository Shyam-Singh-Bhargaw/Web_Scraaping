# python-code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


iris = load_iris()
X = iris.data[:, :2] 
y = iris.target


k = 3  
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', s=150, c='red')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset - K-Means Clustering')
plt.show()

unique_labels, label_counts = np.unique(labels, return_counts=True)
cluster_distribution = dict(zip(unique_labels, label_counts))

print("Cluster Distribution:")
for cluster_label, count in cluster_distribution.items():
    print(f"Cluster {cluster_label}: {count} data points")

print("\nCluster Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i} centroid: {centroid}")
