
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
X=np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
y=np.array([1,2,4,6,8,10,12,14,16,18])

# Create Linear Regression model
model = LinearRegression()

model.fit(X,y)

# Predict house values
y_pred = model.predict(X)

plt.figure()
plt.scatter(X,y,label="Actual data")
plt.plot(X,y_pred)
plt.title("Linear Regression")
plt.xlabel("Independent Variable(X):")
plt.ylabel("Dependent Variable")
plt.legend()
plt.show()