import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('lawyers.csv')

# Extract relevant columns
X_clm = data[['SDY']]  # Independent variable
y_ctm = data['CTM']     # Dependent variable

# Split data into training and test sets (80% train, 20% test)
X_clm_train, X_clm_test, y_ctm_train, y_ctm_test = train_test_split(X_clm, y_ctm, test_size=0.2, random_state=42)

# Train the model
reg_clm = LinearRegression()
reg_clm.fit(X_clm_train, y_ctm_train)

# Predictions
y_ctm_train_pred = reg_clm.predict(X_clm_train)
y_ctm_test_pred = reg_clm.predict(X_clm_test)

# Evaluate the model
train_r2 = r2_score(y_ctm_train, y_ctm_train_pred)
test_r2 = r2_score(y_ctm_test, y_ctm_test_pred)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_clm_train, y_ctm_train, color='blue', label='Training data')
plt.scatter(X_clm_test, y_ctm_test, color='red', label='Test data')
plt.plot(X_clm_train, y_ctm_train_pred, color='black', linewidth=2, label='Fitted line')
plt.xlabel('Number of Sick Days (SDY)')
plt.ylabel('Cases This Month (CTM)')
plt.title('Prediction of CTM from SDY')
plt.legend()
plt.show()

print("Training R-squared:", train_r2)
print("Test R-squared:", test_r2)
