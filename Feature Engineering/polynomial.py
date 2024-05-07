import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('lawyers.csv')

# Extract relevant columns
X = data[['CLM']]  # Independent variable
y = data['CTM']    # Dependent variable

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Calculate R-squared
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training R-squared (Simple Linear Regression):", train_r2)
print("Test R-squared (Simple Linear Regression):", test_r2)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X_train, y_train_pred, color='black', linewidth=2, label='Fitted line')
plt.xlabel('CLM')
plt.ylabel('Cases This Month (CTM)')
plt.title('Prediction of CTM from CLM^2')
plt.legend()
plt.show()
