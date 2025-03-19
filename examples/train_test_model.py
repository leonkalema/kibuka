#!/usr/bin/env python3
"""
Script to train a simple model and save it as a pickle file for testing Kibuka.
"""
import os
import pickle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create output directory
os.makedirs('models', exist_ok=True)

# Load the digits dataset
print("Loading digits dataset...")
digits = load_digits()
X, y = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple random forest model
print("Training random forest classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model
model_path = os.path.join('models', 'digits_classifier.pkl')
print(f"Saving model to {model_path}...")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save some test data for attack examples
test_data_path = os.path.join('models', 'digits_test_data.pkl')
print(f"Saving test data to {test_data_path}...")
test_data = {
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': digits.feature_names if hasattr(digits, 'feature_names') else None,
    'target_names': digits.target_names if hasattr(digits, 'target_names') else None,
    'images': digits.images
}
with open(test_data_path, 'wb') as f:
    pickle.dump(test_data, f)

print("Done! You can now test Kibuka with this model.")
print(f"Example command: python3 kibuka.py --attack model_inversion --model {model_path} --params '{{\"attack_type\": \"gradient_descent\", \"target_class\": 0, \"num_iterations\": 1000}}'")
