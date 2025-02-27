import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "BreastCancer_clean.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=['Id', 'Class'])  # Features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA (use all components to visualize variance)
pca = PCA()
pca.fit(X_scaled)

# Calculate explained variance ratios
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Generate the scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Individual Explained Variance')
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='x', linestyle='--', label='Cumulative Explained Variance')

# Add labels, title, and legend
plt.title('Scree Plot', fontsize=16)
plt.xlabel('Principal Components', fontsize=12)
plt.ylabel('Variance Explained', fontsize=12)
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(fontsize=10)
plt.grid(alpha=0.5)

# Show the plot
plt.show()
