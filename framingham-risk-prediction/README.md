# Cardiovascular Disease Risk Prediction
### Framingham Heart Study — Classification Analysis

Exploratory analysis and predictive modeling to identify the strongest clinical, demographic, and behavioral risk factors associated with 10-year coronary heart disease (CHD) risk, using the landmark Framingham Heart Study dataset.

---

## Overview

Cardiovascular disease is the leading cause of death in the United States, affecting nearly half of all American adults. Early identification of high-risk patients enables targeted prevention — but understanding *which* risk factors matter most, and how well different models can predict risk, remains an active area of clinical research.

This project applies three classification models to the Framingham dataset to predict 10-year CHD risk and identify the most influential predictors.

---

## Dataset

**Source:** [Framingham Heart Study — Kaggle](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)

| Property | Value |
|---|---|
| Records | 4,240 patients |
| Features | 15 (demographic, clinical, behavioral) |
| Target | `TenYearCHD` — binary (1 = developed CHD within 10 years) |
| Class balance | ~85% negative / ~15% positive |

**Key features:** age, sex, smoking status, cigarettes per day, blood pressure medications, prevalent hypertension, diabetes status, total cholesterol, systolic/diastolic BP, BMI, heart rate, glucose

---

## Methods

### Preprocessing
- Median imputation for missing values across 7 columns (glucose had the most: ~9% missing)
- Stratified 80/20 train/test split to preserve class ratio
- StandardScaler applied to continuous features (fit on train only to prevent leakage)

### Models
All models use `class_weight="balanced"` to account for the class imbalance. Performance estimated via 5-fold cross-validation.

| Model | Notes |
|---|---|
| Logistic Regression | Baseline; interpretable coefficients for clinical context |
| Decision Tree | `max_depth=5` to limit overfitting |
| Random Forest | 200 estimators, entropy criterion; feature importances extracted |

### Evaluation Metrics
- Accuracy
- ROC-AUC
- F1 Score

---

## Key Findings

- **Age and systolic blood pressure** were the strongest predictors of 10-year CHD risk across all models
- **Glucose and cigarettes per day** ranked highly in Random Forest feature importances
- **Random Forest** achieved the best AUC and F1, capturing non-linear interactions between features
- **Logistic Regression** performed competitively and remains preferable in clinical settings where interpretability matters

---

## Project Structure

```
framingham-cvd-risk-prediction/
│
├── framingham-cvd-risk-prediction.ipynb   # Main analysis notebook
├── framingham.csv                          # Dataset (from Kaggle link above)
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/framingham-cvd-risk-prediction.git
cd framingham-cvd-risk-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Launch notebook
jupyter notebook framingham-cvd-risk-prediction.ipynb
```

> **Note:** Download `framingham.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression) and place it in the project root before running.

---

## Limitations

- The Framingham cohort is predominantly white and drawn from a single Massachusetts town in the mid-20th century — generalizability to broader populations is limited
- Median imputation for glucose may underrepresent high-risk patients who were less likely to have glucose recorded
- No hyperparameter tuning performed; results represent baseline model behavior

---

## Tools & Libraries

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `seaborn`

---

## Author

**Nyla Morrison**  
MS Data Science — DePaul University  
