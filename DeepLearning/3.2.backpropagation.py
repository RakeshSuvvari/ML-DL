# 🎯 Goal: Visualize Backpropagation in a 2-Layer Neural Network

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate simple binary dataset: XOR problem
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

# Define a small 2-layer neural network
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='sigmoid', name='hidden'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
])

loss_fn1 = tf.keras.losses.MeanSquaredError()
optimizer1 = tf.keras.optimizers.SGD(learning_rate=0.5)

# Training loop with gradient tracking
for epoch in range(10):
    with tf.GradientTape() as tape:
        output = model1(X)
        loss = loss_fn1(y, output)

    # Get gradients for each trainable variable (weights & biases)
    grads = tape.gradient(loss, model1.trainable_variables)
    
    # Apply gradients to update weights
    optimizer1.apply_gradients(zip(grads, model1.trainable_variables))

    # Logging
    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {loss.numpy():.4f}")
    
    for i, (var, grad) in enumerate(zip(model1.trainable_variables, grads)):
        print(f"Layer {i//2 + 1} - {'Weights' if i%2==0 else 'Biases'}")
        print(f"Grad:\n{grad.numpy()}")
        print(f"Updated Param:\n{var.numpy()}")


# 1. 🔁 Animated Plot: See how the decision boundary (or weight space) evolves over epochs
# 2. 💡 ReLU/Softmax Activation + CrossEntropy Loss: Make it more like real-world classification networks

# Build model with ReLU and softmax
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

loss_fn2 = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.1)

# Create meshgrid for decision boundary plotting
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# Initialize figure
fig, ax = plt.subplots()
sc = ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=100, edgecolor='k')
contour = None

def update(epoch):
    global contour
    # One training step
    with tf.GradientTape() as tape:
        logits = model2(X, training=True)
        loss = loss_fn2(y, logits)
    grads = tape.gradient(loss, model2.trainable_variables)
    optimizer2.apply_gradients(zip(grads, model2.trainable_variables))

    # Predict on the meshgrid
    preds = model2.predict(grid, verbose=0)
    Z = np.argmax(preds, axis=1).reshape(xx.shape)

    ax.clear()
    ax.set_title(f"Epoch {epoch+1}")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100, edgecolors='k', label="Data")
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.legend()

ani = FuncAnimation(fig, update, frames=50, interval=500)
plt.tight_layout()
plt.show()