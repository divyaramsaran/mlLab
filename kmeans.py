from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib

# Disable GUI (fixes Tcl/Tk error)
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Load Iris dataset
data = load_iris()
X = data.data

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Get cluster labels (FIXED LINE)
labels = kmeans.labels_

# Plot (first two features)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering on Iris Dataset")
plt.grid()

# Save instead of show (avoids Tkinter)
plt.savefig("kmeans_iris_output.png")

print("K-Means clustering completed!")
print("Plot saved as 'kmeans_iris_output.png'")