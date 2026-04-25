# Step 1: Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# Step 2: Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data # Feature matrix (30 features)
y = data.target # Target labels (0: Malignant, 1: Benign)
# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=42
)
# Step 4: Create Decision Tree classifier
model = DecisionTreeClassifier(
 criterion='gini', # Impurity measure
 max_depth=4 # Limit depth to prevent overfitting
)
# Step 5: Train the model
model.fit(X_train, y_train)
# Step 6: Take dynamic input from user
print("Enter 30 feature values separated by space:")
values = list(map(float, input().split()))
# Step 7: Predict class
prediction = model.predict([values])
# Step 8: Display output
if prediction[0] == 1:
 print("Predicted Class: Benign")
else:
 print("Predicted Class: Malignant")