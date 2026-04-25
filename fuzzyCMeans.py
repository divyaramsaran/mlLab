import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Step 1: Take number of data points
n = int(input("Enter number of data points: "))

# Step 2: Take input points
data = []
print("Enter data points (x y):")
for i in range(n):
    x, y = map(float, input().split())
    data.append([x, y])

# Convert to numpy array and transpose (required format)
data = np.array(data).T

# Step 3: Number of clusters
c = int(input("Enter number of clusters: "))

# Step 4: Apply Fuzzy C-Means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, c=c, m=2, error=0.005, maxiter=1000
)

# Step 5: Membership matrix
print("\nMembership Matrix:")
print(u)

# Step 6: Final cluster assignment (highest membership)
cluster_labels = np.argmax(u, axis=0)
print("\nCluster Assignment:", cluster_labels)

# Step 7: Plot result
plt.scatter(data[0], data[1], c=cluster_labels, cmap='viridis')
plt.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='X')
plt.title("Fuzzy C-Means Clustering (Dynamic Input)")
plt.show()