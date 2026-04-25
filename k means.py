import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 2: Load dataset
data = load_iris()
X = data.data

# Step 3: Take dynamic input for max K
max_k = int(input("Enter maximum value of K: "))

K_values = range(1, max_k + 1)
inertia_values = []

# Step 4: Apply K-Means for each K
for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

    print(f"K = {k}, Sum of Squared Distance = {kmeans.inertia_}")

# Step 5: Plot graph
plt.plot(K_values, inertia_values, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Sum of Squared Euclidean Distance")
plt.title("Elbow Method (Dynamic K)")
plt.grid()
plt.show()

