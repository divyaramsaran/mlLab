import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Step 1: Create Binary Dataset
# -------------------------------
X = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1])

# -------------------------------
# Step 2: Train Logistic Model
# -------------------------------
model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# Step 3: Predict Classes
# -------------------------------
predictions = model.predict(X)
print("Predicted Classes:", predictions)

# -------------------------------
# Step 4: Generate Probability Curve
# -------------------------------
X_test = np.linspace(0,10,100).reshape(-1,1)
probabilities = model.predict_proba(X_test)[:,1]

# -------------------------------
# Step 5: Visualization
# -------------------------------
plt.scatter(X, y, color='red', label='Training Data')
plt.plot(X_test, probabilities, color='blue', label='Probability Curve')

plt.xlabel("X")
plt.ylabel("Probability")
plt.title("Logistic Regression Probability Curve")

plt.legend()
plt.show()