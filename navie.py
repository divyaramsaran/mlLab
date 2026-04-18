from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

model = GaussianNB()

model.fit(X_train, y_train)
print('enter sepalLength, sepalWidth, petalLength, petalWidth')
values = list(map(float, input().split()))
prediction = model.predict([values])
print('predicted class = ', data.target_names[prediction[0]])
