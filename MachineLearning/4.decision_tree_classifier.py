# Goal: Predict whether a person will buy a computer based on Age and Income using simple “yes/no” decision rules.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Prepare the dataset
data = pd.DataFrame({
    'Age': [25, 30, 45, 35, 50],
    'Income': ['High', 'Low', 'High', 'Medium', 'Low'],
    'Buys': ['No', 'Yes', 'Yes', 'Yes', 'No']
})

# Convert text to numeric using label encoding
data['Income'] = data['Income'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Buys'] = data['Buys'].map({'No': 0, 'Yes': 1})

X = data[['Age', 'Income']]
y = data['Buys']

# Step 2: Train Decision Tree
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Step 3: Predict for a new user
prediction = model.predict([[40, 1]])  # Age = 40, Income = Medium
print("Will they buy a computer?", "Yes" if prediction[0] == 1 else "No")

# Step 4: Visualize the decision tree
plt.figure(figsize=(8,6))
plot_tree(model, feature_names=['Age', 'Income'], class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree - Computer Purchase Prediction")
plt.show()
