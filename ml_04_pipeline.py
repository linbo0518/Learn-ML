from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# from sklearn import ensemble
# my_classifier = ensemble.RandomForestClassifier()

# from sklearn import neural_network
# my_classifier = neural_network.MLPClassifier()

my_classifier = my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))