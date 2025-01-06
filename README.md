# Linear Regression from Scratch

## Overview
This project demonstrates how to implement Linear Regression from scratch using Python, without relying on pre-built libraries like `scikit-learn` for the model itself. It showcases fundamental concepts of machine learning, linear algebra, and optimization through gradient descent. By building this project, you'll gain a deeper understanding of the mathematical foundations and mechanics of Linear Regression.

---

## Why This Project?
While libraries like `scikit-learn` simplify the implementation of machine learning models, implementing algorithms from scratch helps to:

- Build a foundational understanding of key concepts such as cost functions, gradient descent, and matrix operations.
- Learn how machine learning models work under the hood.
- Strengthen Python programming skills, especially with `NumPy` for numerical computations.

---

## Prerequisites
To run this project, you need:

- Python 3.7+
- Libraries: `numpy`, `matplotlib`

Install the required libraries with:
```bash
pip install numpy matplotlib
```

---

## Project Steps

### 1. Dataset Creation
We use the `make_regression` function from `scikit-learn` to generate a synthetic dataset. This dataset contains one feature and a target variable with some noise added to simulate real-world conditions.

```python
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
y = y.reshape(-1, 1)  # Reshape for matrix multiplication
```

**Purpose:** Provides a simple dataset to test our Linear Regression implementation.

---

### 2. Adding Bias and Initializing Parameters
We append a bias term (column of ones) to the feature matrix and initialize the parameters randomly.

```python
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
theta = np.random.randn(2, 1)  # Initialize randomly
```

**Purpose:**
- The bias term allows the model to learn the intercept of the line.
- Random initialization of parameters provides a starting point for gradient descent.

---

### 3. Cost Function
The cost function measures the average squared difference between predictions and actual values. It guides the optimization process.

```python
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost
```

**Mathematical Basis:**
\[ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 \]

**Purpose:** Helps evaluate how well the model fits the data.

---

### 4. Gradient Descent
Gradient Descent is an optimization algorithm used to minimize the cost function. It updates the model parameters iteratively to reduce the error.

```python
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        gradients = (1 / m) * (X.T @ (X @ theta - y))
        theta -= learning_rate * gradients
    return theta
```

**Mathematical Basis:**
\[ \theta = \theta - \alpha \cdot \nabla J(\theta) \]
- \( \alpha \): Learning rate (step size).
- \( \nabla J(\theta) = \frac{1}{m} X^T (X \cdot \theta - y) \): Gradient of the cost function.

**Purpose:** Finds the optimal parameters that minimize the cost function.

---

### 5. Training the Model
We use gradient descent to optimize the model parameters.

```python
learning_rate = 0.01
iterations = 1000
theta_opt = gradient_descent(X_b, y, theta, learning_rate, iterations)
```

**Purpose:** Produces the best-fit line for the given dataset.

---

### 6. Visualization
We visualize the dataset and the fitted line to assess the model's performance.

```python
plt.scatter(X, y, color="blue")
plt.plot(X, X_b @ theta_opt, color="red")
plt.title("Linear Regression Fit")
plt.show()
```

**Purpose:** Provides a visual confirmation that the model fits the data correctly.

---

## Results
- The model predicts the target variable by finding the optimal line of best fit.
- The red line in the graph represents the predictions, while the blue points represent the actual data.

---

## Key Concepts Learned
- **Linear Regression Fundamentals:** Understand the relationship between input features and target variables.
- **Gradient Descent:** Optimize parameters to minimize the error.
- **Cost Function:** Quantify how well the model fits the data.
- **Matrix Operations:** Efficiently perform numerical computations using `NumPy`.

---

## How to Run
1. Clone the repository or copy the code into a Python file (`linear_regression.py`).
2. Install the dependencies using `pip install numpy matplotlib`.
3. Run the script:
   ```bash
   python linear_regression.py
   ```
4. View the graph and results.

---

## Future Enhancements
- Extend the implementation to handle multiple features (multivariate regression).
- Add regularization (e.g., Ridge, Lasso) to prevent overfitting.
- Compare with results from `scikit-learn`'s `LinearRegression` for validation.
