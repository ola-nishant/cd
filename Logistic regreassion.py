# Importing libraries
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # Importing train_test_split
import matplotlib.pyplot as plt

# Logistic Regression
class LogisticRegressionCustom:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)

        for i in range(self.iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - Y)) / self.m
            self.theta -= self.learning_rate * gradient

        return self

    def predict(self, X, threshold=0.5):
        predictions = self.sigmoid(np.dot(X, self.theta))
        return (predictions >= threshold).astype(int)

# Function to plot decision boundary
def plot_decision_boundary(X, Y, model, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.RdBu, edgecolors='k')
    plt.title(title)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.show()

# Generating synthetic dataset for simplicity
np.random.seed(0)
X = np.random.randn(100, 2)
Y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Splitting dataset into train and test set
X_train_logistic, X_test_logistic, Y_train_logistic, Y_test_logistic = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Model training using custom Logistic Regression
model_logistic_custom = LogisticRegressionCustom(learning_rate=0.01, iterations=1000)
model_logistic_custom.fit(X_train_logistic, Y_train_logistic)

# Plot decision boundary for custom Logistic Regression
plot_decision_boundary(X_test_logistic, Y_test_logistic, model_logistic_custom, 'Decision Boundary (custom)')
