import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('lawyers.csv')

# Extract relevant columns
X = data[['CLM', 'LVL', 'AGE', 'SDY']]  # Independent variables
y = data['CTM']                         # Dependent variable

# Convert 'LVL' to dummy variables
X = pd.get_dummies(X, columns=['LVL'], drop_first=True)

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

print("Training R-squared:", train_r2)
print("Test R-squared:", test_r2)

# Calculate residuals for training and test data
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# Plot residuals for training data
plt.figure(figsize=(11, 7))
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, residuals_train, color='blue')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM, AGE, LVL and SDY (Training Data)')

# Plot residuals for test data
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, residuals_test, color='red')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CLM, AGE, LVL and SDY (Test Data)')
plt.show()

# Fit the linear regression model
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# Print model summary including AIC and BIC
print(model.summary())
