import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data_scientist.csv')

# Exclude anomalies with IDs 148, 151, and 201 from the dataset
data = data[(data['ID'] != 148) & (data['ID'] != 151) & (data['ID'] != 201)]
data = data.drop(columns=['ID'])

# Feature engineering
data['NormPay'] = data['Pay'] / data['Perf']
data['HapDiff'] = data['EstHap'] - data['EstHap'].mean()

data['Pay*Pay'] = data['Pay'] * data['Pay']
data['Pay*Perf'] = data['Pay'] * data['Perf']
data['Pay*EstHap'] = data['Pay'] * data['EstHap']
data['Perf*Perf'] = data['Perf'] * data['Perf']
data['Perf*EstHap'] = data['Perf'] * data['EstHap']
data['EstHap*EstHap'] = data['EstHap'] * data['EstHap']

data = data.drop(columns=['Pay', 'Perf', 'EstHap'])

# Compute the correlation matrix
corr_matrix = data.corr()
print(corr_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='magma', center=0)
plt.title('Correlation Matrix')
plt.show()