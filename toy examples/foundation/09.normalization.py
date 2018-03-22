from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data_X, data_y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=22,
    n_clusters_per_class=1,
    scale=100)

plt.scatter(data_X[:, 0], data_X[:, 1], c=data_y)
plt.show()

# You can comment the following line to see what is the difference between before and after
data_X = preprocessing.scale(data_X)

X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.1)

clf = SVC()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
