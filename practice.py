# # from collections import Counter
# # data = [1,2,3,4,5,6,6]

# # frequency = Counter(data)
# # max_fre = max(frequency.values())

# # mode = 0
# # for key, values in frequency.items() :
# #     if values == max_fre:
# #         mode = key

# # print(mode)

# from sklearn.datasets import load_diabetes
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import train_test_split

# data = load_diabetes()
# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# model = KNeighborsRegressor(n_neighbors = 5)

# model.fit(X_train, y_train)

# print('enter 10 feature values')
# values = list(map(float, input().split()))
# prediction = model.predict([values])
# print('predicted class: ', prediction[0])


from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeRegressor(max_depth=4)
model.fit(X_train, y_train)

print('enter 8 feature values')
values = list(map(float, input().split()))
prediction = model.predict([values])
print('predicted value = ', prediction[0])