from sklearn import tree

features = [[140, 0], [130, 0], [150, 1], [170, 1]]
labels = ['apple', 'apple', 'orange', 'orange']
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

print(classifier.predict([[160, 1]]))
