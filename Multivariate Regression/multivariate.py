import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('lawyers.csv')

# Extract relevant columns for all iterations
X = data[['CLM']]  # Independent variable for all iterations
y = data['CTM']    # Dependent variable for all iterations

# Split data into training and test sets (80% train, 20% test) for all iterations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iteration 1: Using CLM and AGE
X_iter1 = data[['CLM', 'AGE']]
X_iter1_train = X_iter1.iloc[X_train.index]
X_iter1_test = X_iter1.iloc[X_test.index]

reg_iter1 = LinearRegression()
reg_iter1.fit(X_iter1_train, y_train)

y_iter1_train_pred = reg_iter1.predict(X_iter1_train)
y_iter1_test_pred = reg_iter1.predict(X_iter1_test)

train_r2_iter1 = r2_score(y_train, y_iter1_train_pred)
test_r2_iter1 = r2_score(y_test, y_iter1_test_pred)

# Iteration 2: Using CLM and LVL
X_iter2 = pd.get_dummies(data['LVL'])[['Associate', 'Senior Associate', 'Junior Partner', 'Senior Partner', 'Managing Partner']]
X_iter2_train = X_iter2.iloc[X_train.index]
X_iter2_test = X_iter2.iloc[X_test.index]

reg_iter2 = LinearRegression()
reg_iter2.fit(X_iter2_train, y_train)

y_iter2_train_pred = reg_iter2.predict(X_iter2_train)
y_iter2_test_pred = reg_iter2.predict(X_iter2_test)

train_r2_iter2 = r2_score(y_train, y_iter2_train_pred)
test_r2_iter2 = r2_score(y_test, y_iter2_test_pred)

# Iteration 3: Using CLM and SDY
X_iter3 = data[['CLM', 'SDY']]
X_iter3_train = X_iter3.iloc[X_train.index]
X_iter3_test = X_iter3.iloc[X_test.index]

reg_iter3 = LinearRegression()
reg_iter3.fit(X_iter3_train, y_train)

y_iter3_train_pred = reg_iter3.predict(X_iter3_train)
y_iter3_test_pred = reg_iter3.predict(X_iter3_test)

train_r2_iter3 = r2_score(y_train, y_iter3_train_pred)
test_r2_iter3 = r2_score(y_test, y_iter3_test_pred)

# Calculate residuals for Iteration 1
residuals_iter1_train = y_train - y_iter1_train_pred
residuals_iter1_test = y_test - y_iter1_test_pred

# Plot residuals for Iteration 1
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_iter1_train_pred, residuals_iter1_train, color='blue')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM and AGE (Training Data)')

plt.subplot(1, 2, 2)
plt.scatter(y_iter1_test_pred, residuals_iter1_test, color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM and AGE (Test Data)')
plt.show()

# Calculate residuals for Iteration 2
residuals_iter2_train = y_train - y_iter2_train_pred
residuals_iter2_test = y_test - y_iter2_test_pred

# Plot residuals for Iteration 2
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_iter2_train_pred, residuals_iter2_train, color='blue')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM and LVL (Training Data)')

plt.subplot(1, 2, 2)
plt.scatter(y_iter2_test_pred, residuals_iter2_test, color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM and LVL (Test Data)')
plt.show()

# Calculate residuals for Iteration 3
residuals_iter3_train = y_train - y_iter3_train_pred
residuals_iter3_test = y_test - y_iter3_test_pred

# Plot residuals for Iteration 3
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_iter3_train_pred, residuals_iter3_train, color='blue')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM and SDY (Training Data)')

plt.subplot(1, 2, 2)
plt.scatter(y_iter3_test_pred, residuals_iter3_test, color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM and SDY (Test Data)')
plt.show()

print("Iteration 1:")
print("Training R-squared:", train_r2_iter1)
print("Test R-squared:", test_r2_iter1)

print("\nIteration 2:")
print("Training R-squared:", train_r2_iter2)
print("Test R-squared:", test_r2_iter2)

print("\nIteration 3:")
print("Training R-squared:", train_r2_iter3)
print("Test R-squared:", test_r2_iter3)
