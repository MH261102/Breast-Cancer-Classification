import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# Load the dataset
file_path = "BreastCancer_clean.csv"  # Relative path
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=['Class'])  # Features
y = data['Class']  # Target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensions (for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

# Model parameter grids
param_grids = {
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },
    "logistic_linear": {
        "penalty": ["l2"],
        "C": [0.1, 1, 10, 100],
        "solver": ["lbfgs"]
    },
    "logistic_poly": {
        "penalty": ["l2"],
        "C": [0.1, 1, 10, 100],
        "solver": ["lbfgs"]
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "svm_rbf": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf"]
    },
    "svm_poly": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["poly"]
    }
}

# Initialize models
models = {
    "knn": KNeighborsClassifier(),
    "logistic_linear": LogisticRegression(),
    "logistic_poly": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svm_rbf": SVC(gamma='scale'),
    "svm_poly": SVC(gamma='scale')
}

# Perform grid search for each model
best_results = {}
for model_name, model in models.items():
    print(f"\nStarting Grid Search for {model_name.upper()}...\n")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=7, scoring="accuracy", verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_results[model_name] = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_model": grid_search.best_estimator_
    }
    print(f"Best Parameters for {model_name.upper()}: {grid_search.best_params_}")
    print(f"Best Cross-Validated Accuracy for {model_name.upper()}: {grid_search.best_score_}\n")

# Plot decision boundaries for each model
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

# Evaluate each model and plot
for model_name, result in best_results.items():
    print(f"\nEvaluating {model_name.upper()} on Test Data...\n")
    best_model = result["best_model"]
    y_pred = best_model.predict(X_test)
    test_score = best_model.score(X_test, y_test)
    print(f"Test Accuracy for {model_name.upper()}: {test_score}")
    print(f"Classification Report for {model_name.upper()}:\n")
    print(classification_report(y_test, y_pred))
    plot_decision_boundary(best_model, X_pca, y, f"Decision Boundary for {model_name.upper()} (Best Parameters)")

# Identify the best overall model
final_results = {name: result["best_model"].score(X_test, y_test) for name, result in best_results.items()}
best_model_name = max(final_results, key=final_results.get)
print(f"\nThe best overall model is {best_model_name.upper()} with a test accuracy of {final_results[best_model_name]:.2f}")
