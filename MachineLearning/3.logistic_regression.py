# Goal: Predict whether a student will pass or fail based on their number of study hours.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Step 1: Training data
study_hours = np.array([[1], [2], [3], [4], [5], [6]])
results = np.array([0, 0, 0, 1, 1, 1])  # 1 = Pass, 0 = Fail

# Step 2: Train model
model = LogisticRegression()
model.fit(study_hours, results)

# Step 3: Predict if someone studying 3.5 hours will pass
prediction = model.predict([[3.5]])
probability = model.predict_proba([[3.5]])[0][1]

print(f"Will a student who studies 3.5 hours pass? {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability of passing: {probability:.2f}")

# Step 4: Visualization
x_test = np.linspace(0, 7, 100).reshape(-1, 1)
y_probs = model.predict_proba(x_test)[:, 1]

plt.plot(x_test, y_probs, color='blue', label='Pass Probability')
plt.scatter(study_hours, results, color='red', label='Actual Data')
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Pass/Fail Prediction")
plt.legend()
plt.grid(True)
plt.show()
