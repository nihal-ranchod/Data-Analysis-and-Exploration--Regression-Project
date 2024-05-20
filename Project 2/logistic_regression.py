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
data = data[(data['ID'] != 148) & (data['ID'] != 151) & (data['ID'] != 201)]
data = data.drop(columns=['ID'])


data['NormPay'] = data['Pay'] / data['Perf']
data['HapDiff'] = data['EstHap'] - data['EstHap'].mean()

data['Pay*Pay'] = data['Pay'] * data['Pay']
data['Pay*Perf'] = data['Pay'] * data['Perf']
data['Pay*EstHap'] = data['Pay'] * data['EstHap']
data['Perf*Perf'] = data['Perf'] * data['Perf']
data['Perf*EstHap'] = data['Perf'] * data['EstHap']
data['EstHap*EstHap'] = data['EstHap'] * data['EstHap']
# Selecting features
X = data[['NormPay', 'Pay*Perf', 'HapDiff', 'Pay*Pay', 'Perf']]
y = data['Stay']

# Split data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)  # 0.25 * 0.8 = 0.2

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

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % test_roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Generate evenly spaced values of each feature for plotting
normpay_values = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
pay_perf_values = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100)
hapdiff_values = np.linspace(X_test[:, 2].min(), X_test[:, 2].max(), 100)
pay_pay_values = np.linspace(X_test[:, 3].min(), X_test[:, 3].max(), 100)
perf_values = np.linspace(X_test[:, 4].min(), X_test[:, 4].max(), 100)

# Calculate the corresponding predicted probabilities using the logistic function for each feature
sigmoid_normpay = 1 / (1 + np.exp(-model.coef_[0][0] * normpay_values - model.intercept_[0]))
sigmoid_pay_perf = 1 / (1 + np.exp(-model.coef_[0][1] * pay_perf_values - model.intercept_[0]))
sigmoid_hapdiff = 1 / (1 + np.exp(-model.coef_[0][2] * hapdiff_values - model.intercept_[0]))
sigmoid_pay_pay = 1 / (1 + np.exp(-model.coef_[0][3] * pay_pay_values - model.intercept_[0]))
sigmoid_perf = 1 / (1 + np.exp(-model.coef_[0][4] * perf_values - model.intercept_[0]))

# Plot the sigmoid curves
plt.figure(figsize=(10, 6))
plt.plot(normpay_values, sigmoid_normpay, color='red', label='Logistic Regression (NormPay)')
plt.plot(pay_perf_values, sigmoid_pay_perf, color='blue', label='Logistic Regression (Pay*Perf)')
plt.plot(hapdiff_values, sigmoid_hapdiff, color='green', label='Logistic Regression (HapDiff)')
plt.plot(pay_pay_values, sigmoid_pay_pay, color='orange', label='Logistic Regression (Pay*Pay)')
plt.plot(perf_values, sigmoid_perf, color='purple', label='Logistic Regression (Perf)')

# Plot the actual data points for each feature
plt.scatter(X_test[:, 0], y_test, color='red', label='Actual Data (NormPay)', alpha=0.6)
plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual Data (Pay*Perf)', alpha=0.6)
plt.scatter(X_test[:, 2], y_test, color='green', label='Actual Data (HapDiff)', alpha=0.6)
plt.scatter(X_test[:, 3], y_test, color='orange', label='Actual Data (Pay*Pay)', alpha=0.6)
plt.scatter(X_test[:, 4], y_test, color='purple', label='Actual Data (Perf)', alpha=0.6)

plt.xlabel('Feature Values')
plt.ylabel('Probability')
plt.title('Logistic Regression')
plt.legend()
plt.show()