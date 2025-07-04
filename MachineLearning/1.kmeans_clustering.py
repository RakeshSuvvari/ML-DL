# Goal: Group customers by Spending Score and Age, without knowing their category.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Sample data
data = pd.DataFrame({
    'Age': [22, 45, 23, 40, 25, 30, 38, 27, 50, 31],
    'Spending': [35, 70, 30, 90, 25, 60, 80, 40, 65, 55]
})

# Step 2: Apply KMeans clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)

# Step 3: Get cluster labels
labels = kmeans.labels_

# Step 4: Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Age'], data['Spending'], c=labels, cmap='viridis', s=100)
plt.xlabel("Customer Age")
plt.ylabel("Spending Score")
plt.title("K Means Clustering - Customer Segments")
plt.grid(True)
plt.show()
