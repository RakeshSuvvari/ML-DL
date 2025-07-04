from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Goal: Group customers into clusters without using the Category.
# Same data, but without Category
# Sample data
data = pd.DataFrame({
    'Age': [22, 45, 23, 40, 25],
    'Spending': [35, 70, 30, 90, 25],
    'Category': ['Regular', 'Premium', 'Regular', 'Premium', 'Regular']
})

# Features
X = data[['Age', 'Spending']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.labels_

# Plot clusters
plt.scatter(X['Age'], X['Spending'], c=labels)
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Clusters")
plt.show()
