import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# datasets.make_regression: Generates a random linear regression problem.
# n_samples=100: Generate 100 samples.
# n_features=1: Each sample has 1 feature.
# noise=20: The standard deviation of the Gaussian noise applied to the output.
# random_state=1234: Seed for random number generator for reproducibility.
# X: The generated feature matrix (100 samples, 1 feature each).
# y: The target values (100 values).

X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)  #X is training & testing samples, y is training & test labels

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

print(X_train.shape)    #(80, 1)
print(y_train.shape)    #(80, )

from linear_regression import LinearRegression

regressor = LinearRegression(lr = 0.00001)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def mse(y_true, y_predicted):
      return np.mean((y_true - y_predicted)**2)

mse_value = mse(y_test, predicted)
print(mse_value)


