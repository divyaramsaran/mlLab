# Import libraries
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
# Load dataset
data = load_diabetes()
X = data.data
y = data.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=42
)
# Create KNN regressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)
# User input
print("Enter 10 feature values:")
values = list(map(float, input().split()))
# Prediction
prediction = model.predict([values])
print("Predicted Disease Progression:", prediction[0])