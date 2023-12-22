# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Linear Regression
class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.W) + self.b

# Importing Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

# Selecting sepal length as the independent variable and petal length as the dependent variable
X = data['sepal length (cm)'].values.reshape(-1, 1)
Y = data['petal length (cm)'].values

# Splitting dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Model training
model = LinearRegression(iterations=1000, learning_rate=0.01)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Predicted values ", np.round(Y_pred[:3], 2))
print("Real values ", Y_test[:3])
print("Trained W ", round(model.W[0], 2))
print("Trained b ", round(model.b, 2))

# Visualization on the test set
plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test, Y_pred, color='orange')
plt.title('Petal Length vs Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

# New class for Multiple Linear Regression
class MultipleLinearRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        return X.dot(self.W) + self.b

iris_m = load_iris()
data_m = pd.DataFrame(data=np.c_[iris_m['data'], iris_m['target']],
                       columns=iris_m['feature_names'] + ['target'])

# Selecting sepal length and sepal width as independent variables, and petal length as the dependent variable
X_m = data_m[['sepal length (cm)', 'sepal width (cm)']].values
Y_m = data_m['petal length (cm)'].values

# Splitting dataset into train and test set
X_train_m, X_test_m, Y_train_m, Y_test_m = train_test_split(X_m, Y_m, test_size=0.2, random_state=0)

# Model training using the new class
model_m = MultipleLinearRegression(learning_rate=0.01, iterations=1000)
model_m.fit(X_train_m, Y_train_m)

# Prediction on the test set
Y_pred_m = model_m.predict(X_test_m)
print("Predicted values ", np.round(Y_pred_m[:3], 2))
print("Real values ", Y_test_m[:3])
print("Trained W ", model_m.W)
print("Trained b ", round(model_m.b, 2))

# Visualization on the test set
fig_m = plt.figure(figsize=(12, 5))
ax1_m = fig_m.add_subplot(121, projection='3d')
ax1_m.scatter(X_test_m[:, 0], X_test_m[:, 1], Y_test_m, color='blue')
ax1_m.set_xlabel('Sepal Length (cm)')
ax1_m.set_ylabel('Sepal Width (cm)')
ax1_m.set_zlabel('Petal Length (cm)')
ax1_m.set_title('Actual Petal Length')

ax2_m = fig_m.add_subplot(122, projection='3d')
ax2_m.scatter(X_test_m[:, 0], X_test_m[:, 1], Y_pred_m, color='orange')
ax2_m.set_xlabel('Sepal Length (cm)')
ax2_m.set_ylabel('Sepal Width (cm)')
ax2_m.set_zlabel('Petal Length (cm)')
ax2_m.set_title('Predicted Petal Length')
plt.show()
