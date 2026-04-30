from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=load_iris()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)

print("\n Model Accuracy:",accuracy)

print("Enter sepal length,sepal width,petal length,petal width:")
values=list(map(float,input().split()))
prediction=model.predict([values])
print("\n prediction for sample flower:",data.target_names[prediction[0]])