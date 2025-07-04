# Goal: Classify a new data point based on the majority class of its K closest neighbors.

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Dataset
data = pd.DataFrame({
    'Weight': [120, 150, 200, 180, 100],
    'Size': [7.5, 8.0, 9.0, 8.7, 6.8],
    'Fruit': ['Apple', 'Apple', 'Orange', 'Orange', 'Apple']
})

# Encode labels
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Fruit'])  # Apple=0, Orange=1

X = data[['Weight', 'Size']]
y = data['Label']

# Step 2: Train KNN with k=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Step 3: Predict new fruit
new_fruit = [[130, 7.7]]
prediction = model.predict(new_fruit)
print("Predicted Fruit:", le.inverse_transform(prediction)[0])

# Step 4: Plot decision boundary
x_min, x_max = X['Weight'].min() - 10, X['Weight'].max() + 10
y_min, y_max = X['Size'].min() - 0.5, X['Size'].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 0.05))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X['Weight'], X['Size'], c=y, s=100, edgecolors='k', cmap='coolwarm')
plt.scatter(new_fruit[0][0], new_fruit[0][1], c='green', s=200, marker='X', label='New Prediction')
plt.xlabel("Weight")
plt.ylabel("Size")
plt.title("KNN Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
