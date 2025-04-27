# Admissions-Predictor

Predicting Cambridge Admissions Using LinkedIn and QS Rankings 

## Overview

This project explores whether it’s possible to predict Cambridge admission outcomes for professionals in data-related fields, using only information typically found on a LinkedIn profile. It examines how factors such as university background, time spent in data roles, and leadership experience influence the likelihood of receiving an offer.

This work is submitted as part of my MPhil in Data Intensive Science application.

## Dataset

Source: 

- LinkedIn: Modified publicly available code to scrape through LinkedIn profiles 

- QS dataset: Publicly available data from the QS World University Rankings 2025 report

*Please navigate to the src folder and open 'LinkedIn scraping.ipynb' to learn more about the generation of the admissions dataset*

Features:

- Uni Class: Classification of undergraduate university

- 2025 Rank: QS world ranking of undergraduate university

- leader_months: Duration of leadership roles (in months)

- data_months: Duration of data-related experience (in months)

Target:

- Cambridge?: Binary indicator (1 = admitted, 0 = not admitted)


Minor data cleaning and feature engineering were performed.

As a result of the nature of the Uni Class feature, two datasets were generated:

- dup_df: Original dataset containing duplicate university entries for some applicants.

- arb_df: A cleaned version where one entry per applicant was selected arbitrarily to avoid duplication bias.

## Key Challenges

1. Class imbalance: Significantly more rejected applicants than admitted ones.

2. Small dataset: Risk of overfitting without careful model handling.

3. Potential feature noise: Features like QS rank may introduce variance.

## Approach

1. Data Splitting

    - Hold-out test set reserved before any model fitting.
    
    - Training data further split via cross-validation.
    
2. Preprocessing

    - Oversampling of the minority class (admitted) using SMOTE to balance classes.
    
    - SMOTE applied inside cross-validation folds to avoid data leakage.

3. Baseline Modeling

    - Logistic Regression: Chosen for interpretability and strong baseline benchmarking.

4. Model Development

    - Random Forest Classifier: Tuned using grid search.
    
    - Extra Trees Classifier: Explored to better handle small data and noisy features.

5. Evaluation Metrics

    - Accuracy
    
    - Precision, Recall, F1-score (macro-averaged)
    
    - Training vs. cross-validation performance (to monitor overfitting)

### Why SMOTE?

Chosen over simpler oversampling methods (e.g., random oversampling) because:

- Generates synthetic examples rather than duplicating minority instances.

- Reduces overfitting risk compared to random oversampling.

- Better generalization on unseen data, especially critical with small datasets.

## Results

**On the primary dataset (dup_df):**

|Metric | Random Forest | Extra Trees | Logistic regression |
| --- | --- | --- | --- | 
|Test Accuracy | 0.79 | 0.69 | 0.60 |
|Minority Recall | 0.43 | 0.59 | 0.41 |
|Macro F1-Score | 0.70 | 0.64 | 0.53 |

- Random Forest performed best in terms of test accuracy (0.72) and minority recall (0.74), indicating a strong ability to classify both classes well without significant overfitting.

- Logistic Regression achieved moderate performance but lagged behind in comparison to Random Forest and Extra Trees, especially in terms of minority recall (0.69) and F1-score.

- Extra Trees showed slightly lower accuracy than Random Forest but still maintained a solid performance with a good balance across metrics. It slightly improved minority recall compared to Logistic Regression.


**On the secondary dataset (arb_df):**

|Metric | Random Forest | Extra Trees | Logistic regression |
| --- | --- | --- | --- |
|Test Accuracy | 0.71 | 0.72 | 0.67 |
|Minority Recall | 0.69 | 0.69 | 0.75 |
|Macro F1-Score | 0.71 | 0.72 | 0.67 |

- Extra Trees showed a small improvement in accuracy (0.72) over Random Forest (0.71) and logistic regression (0.67) but did not perform as well as Random Forest in minority recall.

- Logistic Regression outperformed the other models in minority recall (0.75), but the overall accuracy and F1-score were lower compared to the ensemble models.

- The results from arb_df are considered more reliable due to reduced duplication bias, which likely led to more generalizable insights.

Results from arb_df are more reliable due to reduced duplication bias, which led to more generalisable insights.

*Please navigate to the model outputs folder located within the src folder to learn more*

## Tools Used

- Python 3
 
- Jupyter Notebook
 
- scikit-learn (Logistic Regression, Random Forest, Extra Trees)
 
- imbalanced-learn (SMOTE)
 
- pandas, numpy (data manipulation)
 
- matplotlib, seaborn (visualizations)
 
## Conclusion

This project successfully explored the potential of predicting Cambridge admission outcomes for professionals in data-related fields, using only information typically found on a LinkedIn profile. By applying various machine learning models to small, imbalanced datasets, the approach effectively balanced class distribution through resampling techniques like SMOTE. Among the models tested, Extra Trees showed the best performance, improving both recall and F1 scores without overfitting.

While the models demonstrate promising predictive power, the results also underscore the need for continued refinement, particularly in handling feature noise and enhancing model generalization. The findings suggest that, with further data and fine-tuning, such a model could offer valuable insights for admissions prediction in highly competitive fields like data science. This work provides a foundation for future research into scalable, transparent AI applications for educational outcomes.

