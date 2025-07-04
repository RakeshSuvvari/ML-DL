# Goal: Reduce the number of features (dimensions) while preserving as much information (variance) as possible.

# Imagine you have a dataset in 3D, but you want to project it onto 2D for visualization — PCA (Principal Component Analysis) finds the best flat surface to “flatten” your data without losing critical patterns.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target
target_names = iris.target_names

# Step 2: Apply PCA to reduce 4D → 2D
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

# Step 3: Visualize in 2D
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, color, label in zip([0, 1, 2], colors, target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=label, s=100)

plt.title('PCA of IRIS Dataset (2D projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
