# Goal: Predict House Prices based on the Size (sq ft) of the house.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Training data
sizes = np.array([[1000], [1500], [2000], [2500], [3000]])  # Feature (X)
prices = np.array([300, 400, 500, 600, 650])                # Target (y)

# Step 2: Train model
model = LinearRegression()
model.fit(sizes, prices)

# Step 3: Predict price for a new house (e.g., 2200 sq ft)
new_size = np.array([[2200]])
predicted_price = model.predict(new_size)

print(f"Predicted price for 2200 sq ft: ${predicted_price[0]*1000:.2f}")

# Step 4: Plot data and prediction line
plt.scatter(sizes, prices, color='blue', label='Training Data')
plt.plot(sizes, model.predict(sizes), color='red', label='Regression Line')
plt.scatter(new_size, predicted_price, color='green', label='New Prediction')
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.grid(True)
plt.show()
