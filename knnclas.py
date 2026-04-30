# Step 1: Import required libraries
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Step 2: Load the Iris dataset
data = load_iris()
X = data.data # Feature matrix
y = data.target # Target labels

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=42
)

# Step 4: Create KNN classifier with K=5
model = KNeighborsClassifier(n_neighbors=5)
# Step 5: Train the model
model.fit(X_train, y_train)
# Step 6: Take dynamic input from user
print("Enter Sepal Length, Sepal Width, Petal Length, Petal Width:")
values = list(map(float, input().split()))
# Step 7: Predict the class
prediction = model.predict([values])
# Step 8: Display the predicted class name
print("Predicted Class:", data.target_names[prediction[0]])