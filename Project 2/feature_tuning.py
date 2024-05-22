import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from imblearn.over_sampling import SMOTE
import sys

# Redirect output to a text file
output_file = open("model_evaluation_output.txt", "w")
sys.stdout = output_file

# Load the dataset
data = pd.read_csv('data_scientist.csv')

# Exclude anomalies with IDs 148, 151, and 201 from the dataset
data = data[(data['ID'] != 148) & (data['ID'] != 151) & (data['ID'] != 201)]
data = data.drop(columns=['ID'])

# Generating new features
data['NormPay'] = data['Pay'] / data['Perf']
data['HapDiff'] = data['EstHap'] - data['EstHap'].mean()
data['Pay*Pay'] = data['Pay'] * data['Pay']
data['Pay*Perf'] = data['Pay'] * data['Perf']
data['Pay*EstHap'] = data['Pay'] * data['EstHap']
data['Perf*Perf'] = data['Perf'] * data['Perf']
data['Perf*EstHap'] = data['Perf'] * data['Perf']
data['EstHap*EstHap'] = data['EstHap'] * data['EstHap']

# List of all features
features = [
    'Pay', 'Perf', 'EstHap', 'NormPay', 'HapDiff',
    'Pay*Pay', 'Pay*Perf', 'Pay*EstHap',
    'Perf*Perf', 'Perf*EstHap', 'EstHap*EstHap'
]

# Target variable
y = data['Stay']

# Split data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(data[features], y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)  # 0.25 * 0.8 = 0.2

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Function to evaluate a model with a specific combination of features
def evaluate_model(features):
    X_train_selected = X_train_resampled[:, features]
    X_test_selected = X_test[:, features]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_selected, y_train_resampled)
    
    y_test_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_test_pred)
    
    return accuracy

# Variables to track the best combination
best_accuracy = 0
best_combination = None

# Generate all possible combinations of features
for r in range(1, len(features) + 1):
    for combination in combinations(range(len(features)), r):
        accuracy = evaluate_model(combination)
        feature_names = [features[i] for i in combination]
        print(f"Features: {feature_names}, Test Accuracy: {accuracy}")
        
        # Track the best combination
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_combination = combination

# Print the best combination at the end
if best_combination is not None:
    best_feature_names = [features[i] for i in best_combination]
    print(f"\nBest combination of features: {best_feature_names}")
    print(f"Best Test Accuracy: {best_accuracy}")

# Reset stdout to default
sys.stdout = sys.__stdout__
output_file.close()
