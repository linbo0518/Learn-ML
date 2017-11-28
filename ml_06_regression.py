from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

boston_data = datasets.load_boston()

boston_X = boston_data.data
boston_y = boston_data.target

X_train, X_test, y_train, y_test = train_test_split(
    boston_X, boston_y, test_size=0.01)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)
print(y_test)
