import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('data_scientist.csv')

# Exclude anomalies with IDs 148, 151, and 201 from the dataset
filtered_data = data[(data['ID'] != 148) & (data['ID'] != 151) & (data['ID'] != 201)]
filtered_data = filtered_data.drop(columns=['ID'])

# Selecting features
X = filtered_data[['Pay', 'EstHap', 'Perf']]
y = filtered_data['Stay']

# Split data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=kf, scoring='accuracy')

print("Stratified Cross-Validation Scores:", cv_scores)
print("Mean Stratified CV Accuracy:", cv_scores.mean())

# Validate the model with the validation set
y_val_pred = model.predict(X_val)
y_val_pred_prob = model.predict_proba(X_val)[:, 1]
accuracy = accuracy_score(y_val, y_val_pred)
cm = confusion_matrix(y_val, y_val_pred)
report = classification_report(y_val, y_val_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_val_pred_prob)

print("Validation Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)
print("ROC-AUC Score:", roc_auc)

# Test the model with the test set
y_test_pred = model.predict(X_test)
y_test_pred_prob = model.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, y_test_pred)
test_cm = confusion_matrix(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred, zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)

print("Test Accuracy:", test_accuracy)
print("Test Confusion Matrix:\n", test_cm)
print("Test Classification Report:\n", test_report)
print("Test ROC-AUC Score:", test_roc_auc)