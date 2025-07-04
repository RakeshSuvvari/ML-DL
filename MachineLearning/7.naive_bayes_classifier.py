# Goal: Classify text or data based on probabilities learned from feature patterns.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Dataset
data = pd.DataFrame({
    'Text': [
        'Buy cheap watches now',
        'Limited time offer',
        'Meeting schedule tomorrow',
        'Project deadline extended',
        'Earn money quickly',
        'Team lunch today'
    ],
    'Label': ['Spam', 'Spam', 'Ham', 'Ham', 'Spam', 'Ham']
})

# Step 2: Encode labels
data['Label'] = data['Label'].map({'Ham': 0, 'Spam': 1})

# Step 3: Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Text'])

# Step 4: Train Naive Bayes
model = MultinomialNB()
model.fit(X, data['Label'])

# Step 5: Predict a new message
new_message = ['Earn quick money offer']
X_new = vectorizer.transform(new_message)
prediction = model.predict(X_new)
print("Prediction:", "Spam" if prediction[0] == 1 else "Ham")

# Step 6: Visualization — Top Words for Spam vs Ham
feature_names = np.array(vectorizer.get_feature_names_out())
class_labels = ['Ham', 'Spam']
top_n = 5

# Get log probabilities
log_probs = model.feature_log_prob_  # shape: [n_classes, n_features]

# Plot top words for each class
for i, label in enumerate(class_labels):
    top_indices = np.argsort(log_probs[i])[-top_n:]
    top_words = feature_names[top_indices]
    top_scores = log_probs[i][top_indices]

    plt.figure(figsize=(8, 4))
    plt.barh(top_words, top_scores, color='red' if label == 'Spam' else 'blue')
    plt.xlabel("Log Probability")
    plt.title(f"Top {top_n} Words for Class: {label}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
