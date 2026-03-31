# Predicting Type 2 Diabetes Risk Using Machine Learning

A machine learning classification project comparing four models for predicting diabetes diagnosis using the Pima Indians Diabetes Database.

**Course:** Programming for Machine Learning  
**Author:** Nyla Morrison  
**Institution:** DePaul University

---

## Project Overview

Type 2 diabetes is a multifactorial disease, meaning no single variable drives its onset — it emerges from the interaction of genetic, metabolic, and lifestyle factors. This project builds and compares machine learning classifiers to predict diabetes diagnosis from clinical features, and explores whether a heterogeneous ensemble model improves on any individual approach.

---

## Dataset

**Pima Indians Diabetes Database** — 768 patient records, 8 clinical features.

| Feature | Description |
|---|---|
| Times Pregnant | Number of pregnancies |
| Blood Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Fold Thickness | Triceps skinfold thickness (mm) |
| 2-Hour Insulin | 2-hour serum insulin (μU/ml) |
| BMI | Body mass index |
| Family History | Diabetes pedigree function |
| Age | Age in years |

**Target:** `Outcome` — 0 (No Diabetes) / 1 (Diabetes)

---

## Pipeline

### 1. Preprocessing
- Replaced physiologically impossible zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI with column medians
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to address class imbalance (~65% non-diabetic / ~35% diabetic)

### 2. Feature Selection (3-Method Consensus)
Three independent methods were run on the training data; features selected by **at least 2 of 3** methods were kept:
- **Correlation Filter** — absolute correlation with target > 0.05
- **Recursive Feature Elimination (RFE)** — using Logistic Regression as base estimator, top 6 features
- **Random Forest Importance** — top 6 features by importance score

### 3. Models
| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline; features standardized |
| K-Nearest Neighbors (KNN) | k=5; features standardized |
| Random Forest | 200 trees, entropy criterion |
| Support Vector Machine (SVM) | RBF kernel; features standardized |
| **Voting Classifier (Ensemble)** | Soft voting across all 4 models |

### 4. Evaluation
- 5-fold cross-validation on training set
- Final evaluation on held-out test set (80/20 stratified split)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, and **per-class Classification Report**

---

## Results

### Test Set Performance

| Model | Accuracy | F1 (Diabetes) | F1 (No Diabetes) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 71.4% | 0.62 | 0.77 | 0.81 |
| KNN | 70.8% | 0.65 | 0.75 | 0.78 |
| Random Forest | 74.7% | 0.67 | 0.81 | 0.83 |
| SVM | 72.7% | 0.64 | 0.78 | 0.81 |
| **Voting Ensemble** | **75.3%** | **0.68** | **0.80** | **0.83** |

The Voting Ensemble outperformed all individual models on accuracy, Diabetes-class F1, and recall. Per-class classification reports reveal a consistent gap between No Diabetes F1 and Diabetes F1 across all models — a known challenge with this dataset even after SMOTE balancing.

### K-Means Clustering
K-Means (k=2) was applied to explore natural patient subgroups. The resulting clusters showed meaningful alignment with diabetes outcomes, with Blood Glucose and BMI as the primary axes of separation.

---

## Project Structure

```
├── Diabetes_ML_Final.ipynb   # Full notebook with output
├── Pima_Diabetes.csv         # Dataset
└── README.md
```

---

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

---

## Key Takeaways

- Random Forest had the highest test ROC-AUC (0.83) among individual models, consistent with its strength at modeling nonlinear feature interactions
- The Voting Ensemble improved recall for the diabetic class — the most clinically important metric, since a missed diabetes diagnosis (false negative) carries higher risk than a false alarm
- Per-class F1 scores reveal residual majority-class bias that aggregate metrics obscure — a reminder that classification reports are essential for imbalanced medical datasets
