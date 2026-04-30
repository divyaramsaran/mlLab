from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 2: Load dataset
data = load_iris()
X = data.data

# Step 3: Take dynamic input for max K
input = int(input("Enter maximum value of K: "))

K_values = range(1, input + 1)
inertia_values = []

# Step 4: Apply K-Means for each K
for k in K_values:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    inertia_values.append(model.inertia_)
    print(f"K = {k}, Sum of Squared Distance = {model.inertia_}")

# Step 5: Plot graph
plt.plot(K_values, inertia_values)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Sum of Squared Euclidean Distance")
plt.title("Elbow Method (Dynamic K)")
plt.grid()
plt.show()

