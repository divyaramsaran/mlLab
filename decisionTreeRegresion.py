# Step 1: Import required libraries
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
# Step 2: Load California Housing dataset
data = fetch_california_housing()
X = data.data # Feature matrix (8 features)
y = data.target # Target variable (house value)
# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=42
)
# Step 4: Create Decision Tree regression model
model = DecisionTreeRegressor(
 max_depth=4 )# Limit depth to prevent overfitting)
# Step 5: Train the model
model.fit(X_train, y_train)
# Step 6: Take dynamic input
print("Enter 8 feature values separated by space:")
values = list(map(float, input().split()))
# Step 7: Predict house value
prediction = model.predict([values])
# Step 8: Display output
print("Predicted House Value:", prediction[0])