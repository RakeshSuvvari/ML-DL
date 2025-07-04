import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate sample data (points above/below line y = x)
np.random.seed(1)
X = np.random.rand(100, 2)
y = (X[:, 1] > X[:, 0]).astype(int)  # 1 if above y = x, else 0

# Step 2: Perceptron function
def perceptron(X, y, lr=0.1, epochs=20):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        total_error = 0  # Track total error for the epoch
        for i in range(len(X)):
            linear_output = np.dot(X[i], weights) + bias
            prediction = 1 if linear_output > 0 else 0

            # Update rule
            error = y[i] - prediction
            weights += lr * error * X[i]
            bias += lr * error
            total_error += abs(error)

        print(f"Epoch {epoch + 1}/{epochs} — Total Errors: {total_error}")

    return weights, bias


# Step 3: Train the perceptron
weights, bias = perceptron(X, y)

# Step 4: Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=100)

# Plot decision boundary: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
x_vals = np.linspace(0, 1, 100)
y_vals = -(weights[0] * x_vals + bias) / weights[1]
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

plt.title("Perceptron Classification")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()
