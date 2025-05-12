# Admissions-Predictor

### Exploring Career Patterns and Institutional Influence in Data Science

## Overview

This project investigates how educational pedigree and professional experience shape career trajectories in data science, with a focus on distinguishing Cambridge alumni from their peers. Using only publicly available LinkedIn profile data and QS World University Rankings, it examines:

- **Institutional prestige:** How undergraduate university classification and QS rank correlate with later attendance at Cambridge.  
- **Career timing:** How cumulative months in data roles and leadership positions influence the likelihood of pursuing an MPhil at Cambridge.  
- **Skill profiles:** Which technical and interpersonal skills most differentiate Cambridge graduates from other data professionals.

This analysis supports my application to the MPhil in Data Intensive Science at the University of Cambridge by demonstrating my ability to design, implement, and interpret data-science workflows end-to-end.

## Data Sources

- **LinkedIn profiles:** Scraped via a custom adaptation of the StaffSpy tool (see `src/LinkedIn scraping.ipynb`).  
- **QS 2025 Rankings:** Publicly available global university rankings, matched to each profile’s listed institutions.

## Key Features

- **Uni Class:** Categorical prestige tier (Oxbridge, Russell Group, Other)  
- **2025 Rank:** Numeric QS university ranking  
- **data_months:** Total months in data-related roles  
- **leader_months:** Total months in leadership roles  
- **Eight validated skills:** One-hot indicators for skills (e.g. SQL, Excel, Physics) shown to most strongly associate with Cambridge attendance  

## Target

- **Cambridge?** (1 = alumni, 0 = non-alumni)

## Datasets

- **dup_df:** “Duplication” method splitting each profile into two rows (one per listed university) to maximize data diversity.  
- **arb_df:** “Arbitrary” method selecting one university per profile for a more conservative, noise-reduced dataset.

## Challenges

1. **Class imbalance:** Far fewer Cambridge alumni than non-alumni.  
2. **Limited sample size:** Heightened risk of overfitting. 
3. **Feature noise:** Inconsistencies and errors introduced by free-text LinkedIn fields (e.g. variable skill labels, missing dates) and by approximating QS rank bands (range midpoints and manual overrides).
4. **Network sampling bias:** Because scraping was limited to my own LinkedIn connections, the dataset likely over-represents profiles similar to my background (e.g. STEM disciplines), which may skew the validated skills and model generalizability.

## Methodology

1. **Train-test split** with a held-out 20% test set.  
2. **SMOTE oversampling** inside 5-fold cross-validation to balance classes.  
3. **Baseline & Ensemble models:**  
   - Logistic Regression (interpretable baseline)  
   - Random Forest & Extra Trees (bagged tree ensembles)  
   - XGBoost (gradient boosted trees)  
4. **Hyperparameter tuning** via `GridSearchCV` optimizing F1-score.  
5. **Evaluation:** Accuracy, precision/recall/F1 (macro-averaged), ROC-AUC, and cross-validation scores.  
6. **Interpretability:** SHAP summary plots to reveal feature impacts.

## Results (Highlights)

### Duplication Dataset (dup_df) Results

| Model               | Accuracy | ROC-AUC | Recall (on admits) | F1 (on admits) |
| ------------------- | -------: | ------: | -----------------: | -------------: |
| XGBoost             |   0.7569 |  0.7057 |               0.59 |           0.58 |
| Random Forest       |   0.7514 |  0.6661 |               0.47 |           0.52 |
| Logistic Regression |   0.6630 |  0.6105 |               0.49 |           0.45 |
| Extra Trees         |   0.7569 |  0.6759 |               0.49 |           0.53 |

### Arbitrary Dataset (arb_df) Results

| Model               | Accuracy | ROC-AUC | Recall (on admits) | F1 (on admits) |
| ------------------- | -------: | ------: | -----------------: | -------------: |
| XGBoost             |   0.6695 |  0.6678 |               0.65 |           0.64 |
| Random Forest       |   0.6271 |  0.6198 |               0.56 |           0.57 |
| Logistic Regression |   0.6610 |  0.6664 |               0.71 |           0.65 |
| Extra Trees         |   0.6441 |  0.6390 |               0.60 |           0.60 |


Across both dataset variants, our ensemble models achieved strong overall accuracy, but the story changes when we look closer:

- On the duplication set, XGBoost and Extra Trees tied for the highest accuracy (0.7569), indicating that simply bagging randomized trees can match gradient boosting when noise is amplified by duplicating profiles.

- On the arbitrary set, Logistic Regression led the way in admit‐class recall (0.71), suggesting that in a cleaner, less‐noisy dataset a linear decision boundary more effectively isolates Cambridge alumni.

- XGBoost’s performance shift—accuracy dropping but recall rising when moving from dup_df to arb_df—highlights how the choice of dataset shaping can drastically alter which model metrics look best, and thus which model you’d select depending on whether you prioritize catching all true admits or maximizing overall correctness.

### SHAP Analysis Results

On closer analysis of our SHAP summary plots we learn:

- Prestige signals shift depending on dataset construction—categorical class dominates in the noisy duplication set, while fine-grained QS rank takes over once noise is reduced.

- Length of data experience becomes a strong positive predictor only after reducing duplication bias, suggesting careful data curation can clarify true career–admissions relationships.

- High SQL emphasis surprisingly decreases predicted Cambridge likelihood, hinting at a deeper career‐stage or skill‐focus distinction between research‐oriented applicants and seasoned industry specialists.

> _Full tables, plots, and detailed outputs are in the `src/model_outputs/` and `results/` folders._


## Conclusion

This work demonstrates a reproducible, end-to-end data-science pipeline—from web-scraped profiles to model interpretation—highlighting how institutional and career factors intersect in elite academic admissions. It lays the groundwork for richer feature exploration and application-specific threshold tuning, with potential for broader validation against real-world admissions data.

