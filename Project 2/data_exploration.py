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

# Correlation matrix without the 'ID' column
correlation_matrix = data.drop(columns=['ID']).corr()

# Change the color palette and center the color mapping at 0
sns.heatmap(correlation_matrix, annot=True, cmap='magma', center=0, vmin=-1, vmax=1)

# Show the plot
plt.show()

# Customizing seaborn style
sns.set(style="whitegrid")

# Filter the dataset based on the "Stay" column
stay_data = data[data['Stay'] == 1]  # Data for employees who stay
leave_data = data[data['Stay'] == 0]  # Data for employees who leave

combined_data = pd.concat([stay_data, leave_data])

# Pair plot for both stay and leave groups
sns.pairplot(combined_data.drop(columns=['ID']), hue='Stay', palette='husl')
plt.suptitle('Pair Plot for Employees Who Stay(1) and Leave(0)', y=1)  # Title with some space above the plot
plt.show()