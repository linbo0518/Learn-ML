from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
    n_samples=100, n_features=1, n_targets=1, noise=1)
noise_X, noise_y = datasets.make_regression(
    n_samples=100, n_features=1, n_targets=1, noise=10)

plt.scatter(X, y)
plt.show()

plt.scatter(noise_X, noise_y)
plt.show()
