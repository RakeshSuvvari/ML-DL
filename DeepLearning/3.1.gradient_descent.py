import numpy as np
import matplotlib.pyplot as plt

# Simple quadratic loss function: L(w) = (w - 3)^2
def loss(w): return (w - 3)**2
def gradient(w): return 2 * (w - 3)

# Initialize
w = 0
learning_rate = 0.1
weights = []
losses = []

# Gradient Descent Loop
for i in range(20):
    weights.append(w)
    losses.append(loss(w))
    w = w - learning_rate * gradient(w)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(weights, losses, 'o-', label="Loss curve")
plt.xlabel("Weight")
plt.ylabel("Loss")
plt.title("Gradient Descent Minimizing Loss")
plt.grid(True)
plt.legend()
plt.show()
