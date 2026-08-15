# Adult Income Classification - ML Assignment 2
# BITS Pilani M.Tech in AI/ML Semester 1

## a. Problem Statement
Predicting whether a person earns more than $50K/year based on census demographic and employment data. 
This is a binary classification problem (income >$50K: Yes/No), solved and compared using five machine learning models:
Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Random Forest (Ensemble).
This dataset provides a real-world example of binary classification with a mix of numerical and categorical features, and has practical relevance for socioeconomic analysis and resource allocation.

## b. Dataset Description
- **Source:** UCI Machine Learning Repository / Kaggle - Adult Income (Census Income) Dataset
- **Instances:** 30,162 (after removing rows with missing/null values from the raw 32,561)
- **Features:** 14 attributes (>12 minimum required) including target
- **Target variable:** income (binary: <=50K or >50K)
- **Task type:** Binary Classification

## c. GitHub Repository Link
[https://github.com/chocoblin/adult-income-classifier](https://github.com/chocoblin/adult-income-classifier)

## d. Comparison Table (Tuned Models)

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7916 | 0.8039 | 0.6553 | 0.3020 | 0.4134 | 0.3413 |
| Decision Tree (tuned) | 0.8440 | 0.8862 | 0.7036 | 0.6196 | 0.6589 | 0.5602 |
| KNN (tuned) | 0.8165 | 0.8556 | 0.6400 | 0.5610 | 0.5979 | 0.4814 |
| Naive Bayes | 0.7858 | 0.8259 | 0.6308 | 0.2877 | 0.3951 | 0.3191 |
| Random Forest (tuned) | 0.8628 | 0.9119 | 0.7742 | 0.6149 | 0.6854 | 0.6056 |

*Note: Baseline (default hyperparameter) results and comparison also available
in `model/train_models.ipynb` — hyperparameter tuning via GridSearchCV
improved Decision Tree AUC by +0.136 and MCC by +0.064, the largest gain of
any model, by addressing overfitting from unconstrained tree depth.*