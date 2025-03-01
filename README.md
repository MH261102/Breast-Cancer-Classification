# 🎯 Breast Cancer Classification Project

## 📋 Overview
This project implements various machine learning models to classify breast cancer cases using cell image features. The goal is to compare different classification approaches and identify the most effective model for early detection.

## 🗂️ Project Structure
```
📁 Project Root
├── 📁 Lab
│   ├── 📊 BreastCancer_clean.csv
│   ├── 📊 BreastCancer_PCA_Clusters.csv
│   ├── 📊 BreastCancer.csv
│   ├── 🐍 data_cleanup.py
│   ├── 🐍 logreg.py
│   └── 🐍 main.py
├── 📁 Lecture Notes
├── 📁 Pictures
│   ├── 📈 KNN.png
│   ├── 📈 LOG-LINEAR.png
│   ├── 📈 POLY_SVM.png
│   ├── 📈 RBF_SVM.png
│   └── 📈 RF.png
├── 📄 README.md
└── 📄 Report.docx
```

## 🤖 Models Implemented
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
    - Linear Kernel
    - Polynomial Kernel
    - RBF Kernel
- Random Forest

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 🚀 Usage

### Data Preprocessing
```python
python Lab/data_cleanup.py
```
- Handles missing values
- Performs feature scaling
- Applies PCA for dimensionality reduction

### Running Models
```python
python Lab/main.py
```

### Logistic Regression Analysis
```python
python Lab/logreg.py
```

## 📊 Results
Model performance metrics are available in the Pictures directory:
- Accuracy scores
- ROC curves
- Confusion matrices
- Feature importance plots

## 📝 Contributing
Feel free to submit issues and enhancement requests.

## 📜 License
This project is licensed under the MIT License.

## 👥 Authors
- Mihir Hurwanth

## 📧 Contact
- Email: mhurwanth@gmail.com
