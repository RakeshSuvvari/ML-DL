# Goal: SVM tries to find the best boundary (hyperplane) that separates two classes with the maximum margin.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

# Step 1: Generate synthetic data
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.03,
    class_sep=1.5,
    random_state=42
)

# Step 2: Train SVM
model = SVC(kernel='linear')  # You can try 'rbf', 'poly', etc.
model.fit(X, y)

# Step 3: Predict a new point
new_point = np.array([[1.5, -0.5]])
prediction = model.predict(new_point)
print("Predicted class for new point:", prediction[0])

# Step 4: Visualize decision boundary
disp = DecisionBoundaryDisplay.from_estimator(
    model, X, response_method="predict", cmap="coolwarm", alpha=0.4
)

plt.figure(figsize=(8, 6))
disp.plot()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=100)
plt.scatter(new_point[0][0], new_point[0][1], c='green', marker='X', s=200, label="New Point")
plt.title("SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
