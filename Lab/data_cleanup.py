import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Load dataset
df = pd.read_csv('BreastCancer.csv')

# Set the 'Id' column as the index for easier referencing
df.set_index('Id', inplace=True)

# Map the 'Class' column to binary values for ML purposes
df['Class'] = df['Class'].map({'benign': 0, 'malignant': 1})

# Define the continuous variables
continuous_variables = [
    'Cl.thickness', 'Cell.size', 'Cell.shape', 'Marg.adhesion', 
    'Epith.c.size', 'Bare.nuclei', 'Bl.cromatin', 'Normal.nucleoli', 'Mitoses'
]

# Impute missing values using KNN imputation
imputer = KNNImputer(n_neighbors=5)
df[continuous_variables] = imputer.fit_transform(df[continuous_variables])

# Standardize continuous variables to ensure equal contribution to the model
scaler = StandardScaler()
df[continuous_variables] = scaler.fit_transform(df[continuous_variables])

# Save the cleaned dataset
df.to_csv('BreastCancer_clean.csv', index=True)



