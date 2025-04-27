# Admissions-Predictor
Predicting Cambridge Admissions Using LinkedIn and QS Rankings 

## Overview
This project builds a machine learning pipeline to predict Cambridge admission outcomes based on applicant background data.
It addresses challenges of small sample size, class imbalance, and feature noise through careful preprocessing and model tuning.

### Dataset
Features: Uni Class, 2025 Rank, leader_months, data_months

Target: Cambridge? (1 = admitted, 0 = not admitted)

### Methodology
Train/Test Split before any resampling to prevent leakage.

SMOTE applied inside cross-validation to balance classes.

### Models

Logistic Regression (baseline)

Random Forest (tuned)

Extra Trees (tuned, better handling of noise)

Hyperparameter Tuning with GridSearchCV.

### Key Results

| Dataset | Model | Accuracy | Minority Recall | Macro F1-Score|
| --- | --- | --- | --- | --- |
| dup_df | Random Forest | 0.69 | 0.53 | 0.62|
| dup_df | Extra Trees | 0.69 | 0.59 | 0.64|
| arb_df | Random Forest | 0.69 | 0.63 | 0.68|
| arb_df | Extra Trees | 0.72 | 0.69 | 0.72|


Extra Trees improved recall and F1-scores without overfitting.
 
### Tools
Python (scikit-learn, imbalanced-learn, pandas, numpy)

