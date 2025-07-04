# Goal: Make predictions using many decision trees and combine their results to reduce overfitting and improve accuracy.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Create a larger dataset
data = pd.DataFrame({
    'Age': [25, 30, 45, 35, 50, 29, 41, 38, 60, 27],
    'Income': ['High', 'Low', 'High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High'],
    'Buys': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
})

# Encode categorical variables
le_income = LabelEncoder()
data['Income'] = le_income.fit_transform(data['Income'])  # High=0, Low=1, Medium=2
data['Buys'] = data['Buys'].map({'No': 0, 'Yes': 1})

X = data[['Age', 'Income']]
y = data['Buys']

# Step 2: Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 3: Predict for a new customer
new_customer = [[40, le_income.transform(['Medium'])[0]]]
prediction = model.predict(new_customer)
print("Will they buy a computer?", "Yes" if prediction[0] == 1 else "No")

# Step 4: Plot decision boundaries
x_min, x_max = X['Age'].min() - 5, X['Age'].max() + 5
y_min, y_max = X['Income'].min() - 1, X['Income'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 0.1))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X['Age'], X['Income'], c=y, s=100, edgecolors='k', cmap='coolwarm')
plt.scatter(new_customer[0][0], new_customer[0][1], c='green', s=200, marker='X', label='New Prediction')
plt.xlabel("Age")
plt.ylabel("Income (encoded)")
plt.title("Random Forest Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
