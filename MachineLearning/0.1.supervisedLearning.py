from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Goal: Predict the Category (Regular or Premium) using Age and Spending Score.
# Sample data
data = pd.DataFrame({
    'Age': [22, 45, 23, 40, 25],
    'Spending': [35, 70, 30, 90, 25],
    'Category': ['Regular', 'Premium', 'Regular', 'Premium', 'Regular']
})

# Features and labels
X = data[['Age', 'Spending']]
y = data['Category']

# Train a decision tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for a new customer
print(model.predict([[30, 60]]))  # Predicts category
