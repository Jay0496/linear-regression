from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
y = y.reshape(-1, 1)  # Reshape for matrix multiplication

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        gradients = (1 / m) * (X.T @ (X @ theta - y))
        theta -= learning_rate * gradients
    return theta

X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
theta = np.random.randn(2, 1)  # Initialize randomly

learning_rate = 0.01
iterations = 1000
theta_opt = gradient_descent(X_b, y, theta, learning_rate, iterations)


plt.scatter(X, y, color="blue")
plt.plot(X, X_b @ theta_opt, color="red")
plt.title("Linear Regression Fit")
plt.show()
