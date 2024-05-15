import pandas as pd

# Load the dataset
data = pd.read_csv('data_scientist.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Correlation matrix
print(data.corr())

# Scatter plots for numerical variables
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data)
plt.show()