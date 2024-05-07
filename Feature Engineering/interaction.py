import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('lawyers.csv')

# Encoding the LVL column
ordinal_encoder = OrdinalEncoder(categories=[['Associate', 'Senior Associate', 'Junior Partner', 'Senior Partner', 'Managing Partner']])
data['LVL_encoded'] = ordinal_encoder.fit_transform(data[['LVL']])

# Extract relevant columns
X = data[['CLM', 'LVL_encoded']]  # Independent variables
y = data['CTM']                   # Dependent variable

# Create feature interaction between CLM and LVL_encoded
X['CLM_LVL_interaction'] = X['CLM'] * X['LVL_encoded']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
reg = LinearRegression()
reg.fit(X_train[['CLM_LVL_interaction']], y_train)

# Predictions
y_train_pred = reg.predict(X_train[['CLM_LVL_interaction']])
y_test_pred = reg.predict(X_test[['CLM_LVL_interaction']])

# Evaluate the model
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Plot the interaction between CLM and LVL with the fitted line
plt.figure(figsize=(10, 6))
plt.scatter(X_train['CLM_LVL_interaction'], y_train, color='blue', label='Training data')
plt.scatter(X_test['CLM_LVL_interaction'], y_test, color='red', label='Test data')
plt.plot(X_train[['CLM_LVL_interaction']], y_train_pred, color='black', linewidth=2, label='Fitted line')
plt.xlabel('CLM * LVL_encoded')
plt.ylabel('Cases This Month (CTM)')
plt.title('Interaction of CLM and LVL')
plt.legend()
plt.show()

print("Training R-squared:", train_r2)
print("Test R-squared:", test_r2)
