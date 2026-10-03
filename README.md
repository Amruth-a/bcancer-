# bcancer-
# Breast Cancer Classification

## Project Overview

This project uses machine learning to classify breast tumors as **Benign** or **Malignant** using diagnostic measurements.

The project follows a complete machine learning workflow:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Feature analysis and correlation
* Feature selection
* Model training
* Model comparison
* Final model evaluation
* Unsupervised K-Means clustering
* PCA visualization

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset.

* Samples: **569**
* Features: **30 numerical features**
* Target: `diagnosis`
* `B` = Benign
* `M` = Malignant

Dataset source: UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Data Cleaning

The original dataset contained an unnecessary `Unnamed: 32` column containing only missing values.

This column was removed before analysis.

The `id` column was also excluded from model training because it is an identifier rather than a meaningful predictive feature.

The dataset did not contain duplicate rows after inspection.

## Exploratory Data Analysis

EDA was performed to understand:

* Class distribution
* Feature distributions
* Correlations between features
* Differences in feature means between benign and malignant cases
* Feature overlap between the two classes

A correlation heatmap was used to identify relationships between numerical features.

Several measurements such as radius, perimeter, area, concavity, and concave points showed strong relationships with the diagnosis.

## Feature Selection

Initially, all 30 numerical features were used.

Feature selection was then performed using **Recursive Feature Elimination (RFE)** with Logistic Regression.

RFE selected 10 features:

* `concave points_mean`
* `radius_se`
* `area_se`
* `compactness_se`
* `radius_worst`
* `texture_worst`
* `perimeter_worst`
* `area_worst`
* `concavity_worst`
* `concave points_worst`

The selected features were standardized before training the final models.

## Models

The following supervised learning models were explored:

### Logistic Regression

Accuracy after RFE feature selection:

**97.37%**

### Support Vector Machine

An SVM with an **RBF kernel** was trained using the selected features.

Accuracy:

**98.25%**

ROC-AUC:

**≈ 0.99**

### Random Forest

A Random Forest classifier was also trained using the selected features.

Accuracy:

**≈ 97%**

## Final Model Evaluation

The final selected supervised model was the **RBF SVM**.

### Test Set Results

| Metric              |     Result |
| ------------------- | ---------: |
| Accuracy            | **98.25%** |
| Benign Precision    |    **97%** |
| Benign Recall       |   **100%** |
| Benign F1-score     |    **99%** |
| Malignant Precision |   **100%** |
| Malignant Recall    |    **95%** |
| Malignant F1-score  |    **98%** |
| ROC-AUC             | **≈ 0.99** |

The test set contained **114 samples**.

### Confusion Matrix

```text
                 Predicted
              Benign  Malignant
Actual Benign     72       0
       Malignant   2      40
```

The model correctly classified 112 of the 114 test samples.

There were **2 false negatives**, where malignant cases were classified as benign.

## Unsupervised Learning

K-Means clustering was also explored as an unsupervised learning experiment.

The model was trained without using the diagnosis labels.

The resulting clusters were compared with the known diagnosis labels afterward to understand how well the natural grouping corresponded to benign and malignant cases.

PCA was used to reduce the selected feature space to two dimensions for visualization of the clusters and model decision regions.

## Key Learning Outcomes

Through this project, I practiced:

* Data cleaning with Pandas
* Exploratory data analysis
* Correlation analysis
* Feature selection using RFE
* Feature scaling
* Logistic Regression
* SVM with RBF kernel
* Random Forest classification
* K-Means clustering
* PCA
* Confusion matrices
* Precision, recall and F1-score
* ROC curves and ROC-AUC
* Comparing supervised and unsupervised learning

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## Project Structure

```text
breast-cancer-classification/
│
├── breast_cancer_classification.ipynb
└── README.md
```

## Conclusion

This project demonstrated a complete machine learning classification workflow, from data exploration and feature selection to model training and final evaluation.

The RBF SVM achieved **98.25% test accuracy** and approximately **0.99 ROC-AUC** on the held-out test set.
