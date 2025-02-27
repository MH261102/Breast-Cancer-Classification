import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA

# Load the dataset
file_path = "BreastCancer_clean.csv"  # Relative path
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(column=['Id', 'Class'])  # Features
y = data['Class']  # Target

# Apply PCA to reduce dimensions (for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, stratify=y)

# Parameter grid for SVM with polynomial kernel
param_grid = {
    "C": [0.002, 0.004, 0.006, 0.008],  # Regularization parameter
    "degree": [2, 3, 4, 5],        # Polynomial degree
    "coef0": [0, 0.5, 1, 2],       # Independent term in polynomial kernel
    "gamma": ["scale", "auto"]     # Kernel coefficient
}

# Initialize SVM model with polynomial kernel
svm_poly = SVC(kernel='poly')

# Perform grid search
print("\nStarting Grid Search for SVM with Polynomial Kernel...\n")
grid_search = GridSearchCV(estimator=svm_poly, param_grid=param_grid, cv=7, scoring="accuracy", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Store the best results
best_svm_model = grid_search.best_estimator_
print(f"Best Parameters for SVM (Polynomial Kernel): {grid_search.best_params_}")
print(f"Best Cross-Validated Accuracy for SVM (Polynomial Kernel): {grid_search.best_score_:.4f}\n")

# Evaluate the best SVM model on the test data
print("\nEvaluating SVM with Polynomial Kernel on Test Data...\n")
y_pred = best_svm_model.predict(X_test)
test_score = best_svm_model.score(X_test, y_test)
print(f"Test Accuracy for SVM (Polynomial Kernel): {test_score:.4f}\n")

# Generate a formatted classification report
def formatted_classification_report(y_true, y_pred, target_names=None):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    accuracy = (y_true == y_pred).mean()

    # Create a DataFrame for better formatting
    metrics_table = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support
    }, index=target_names if target_names else [f"Class {i}" for i in range(len(precision))])

    # Add overall accuracy as an additional row
    metrics_table.loc["Accuracy"] = [accuracy, accuracy, accuracy, sum(support)]

    # Format numbers to 4 decimal places
    return metrics_table.round(4)

classification_table = formatted_classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
print("\nClassification Report (4 Decimal Places):\n")
print(classification_table)

# Plot decision boundary for the SVM model
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

plot_decision_boundary(best_svm_model, X_pca, y, "Decision Boundary for SVM (Polynomial Kernel)")
