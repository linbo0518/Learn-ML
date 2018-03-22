import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_iris_data = [0, 50, 100]

train_target = np.delete(iris.target, test_iris_data)
train_data = np.delete(iris.data, test_iris_data, axis=0)

test_targrt = iris.target[test_iris_data]
test_data = iris.data[test_iris_data]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print(test_targrt)
print(clf.predict(test_data))

import graphviz
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)
graph.save(filename='iris')