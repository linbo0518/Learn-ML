from sklearn.learning_curve import learning_curve
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
X = digits.data
y = digits.target

train_size, train_loss, test_loss = learning_curve(
    SVC(gamma=0.01),  # overfitting, set gamma=0.001 will be better 
    X,
    y,
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1],
    cv=10,
    scoring='mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_size, train_loss_mean, 'o-', color='r', label='train')
plt.plot(train_size, test_loss_mean, 'o-', color='g', label='cross validation')

plt.xlabel('training examples')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
