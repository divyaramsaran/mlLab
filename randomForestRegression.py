from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data=fetch_california_housing()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
print("Mean squared error:",mse)