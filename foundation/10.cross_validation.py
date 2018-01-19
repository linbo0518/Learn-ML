from sklearn import datasets
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
print(KNN.score(X_test, y_test))

k_range = range(1, 31)
k_score = []
for k in k_range:
    KNN = KNeighborsClassifier(n_neighbors=k)

    # for classification
    score = cross_val_score(KNN, X, y, cv=10, scoring='accuracy')

    # for regression
    # loss = -cross_val_score(KNN, X, y, cv=10, scoring='mean_squared_error')

    k_score.append(score.mean())

plt.plot(k_range, k_score)
plt.xlabel("K")
plt.ylabel("cross-validated accuracy")
plt.show()