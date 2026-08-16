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

## d. Models used
- Logistic Regression
- Naive Bayes
- Decision Tree (tuned)
- K-Nearest Neighbours (tuned)
- Random Forest (tuned)

*Note: Baseline (default hyperparameter) results and comparison also available in `model/train_models.ipynb` for before/after comparison — hyperparameter tuning via GridSearchCV improved Decision Tree AUC by +0.136 and MCC by +0.064, the largest gain of any model, by addressing overfitting from unconstrained tree depth.*

### Hyperparameter Tuning
Logistic Regression and Naive Bayes have few/no meaningful hyperparameters to tune for this dataset, so tuning was focused on Decision Tree, KNN, and Random Forest using `GridSearchCV` with 3-fold cross-validation, optimizing for **F1-score** (chosen over accuracy due to the ~75/25 class imbalance in the target variable).

**Best parameters found:**
- Random Forest: `n_estimators=250, max_depth=18, min_samples_split=2`
- KNN: `n_neighbors=7`
- Decision Tree: `max_depth=12, min_samples_split=15`

### Comparison Table (final, tuned where applicable)

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7916 | 0.8039 | 0.6553 | 0.3020 | 0.4134 | 0.3413 |
| Decision Tree (tuned) | 0.8440 | 0.8862 | 0.7036 | 0.6196 | 0.6589 | 0.5602 |
| KNN (tuned) | 0.8165 | 0.8556 | 0.6400 | 0.5610 | 0.5979 | 0.4814 |
| Naive Bayes | 0.7858 | 0.8259 | 0.6308 | 0.2877 | 0.3951 | 0.3191 |
| Random Forest (tuned) | 0.8628 | 0.9119 | 0.7742 | 0.6149 | 0.6854 | 0.6056 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Decent accuracy (79.2%) but low recall (0.30) — struggles to identify actual >$50K earners due to the ~75/25 class imbalance and its linear decision boundary. LabelEncoded categorical features (which have no true ordinal relationship) further limit its ability to model complex interactions. |
| Decision Tree | After tuning (`max_depth=12`, `min_samples_split=15`), AUC jumped from 0.75 to 0.886 and MCC from 0.496 to 0.560 — the baseline tree was overfitting the training data with unconstrained depth, and tuning corrected this, bringing performance close to the ensemble model. |
| kNN | Strong AUC (0.856 tuned) indicating good probability ranking, but sensitive to LabelEncoded high-cardinality features (e.g. native-country) since Euclidean distance treats arbitrary integer categories as meaningfully "close" or "far." `n_neighbors=7` was found optimal via grid search. |
| Naive Bayes | Highest AUC (0.826) among non-ensemble, non-tree models, but poor accuracy/recall (0.29) — its default 0.5 threshold is poorly calibrated for this imbalanced, largely categorical dataset, since Gaussian NB assumes continuous normally-distributed features, which doesn't hold for LabelEncoded categories. |
| Random Forest (Ensemble) | Best performer across every metric (Accuracy 0.863, AUC 0.912, MCC 0.606). Averaging across many trees reduces the overfitting seen in the single Decision Tree while handling the LabelEncoded categorical features and class imbalance more robustly than the other four models. |
| **Overall Winner** | **Random Forest (tuned, Ensemble)** — highest scores on all 6 metrics, most robust against class imbalance and LabelEncoding limitations. |

### In Plain English

- **Random Forest wins because it's a committee, not a lone genius.**
    ~250 trees vote instead of one tree overfitting on its own quirks.

- **LR and Naive Bayes play it safe.** 
    Since ~75% of people earn ≤50K, guessing "≤50K" a lot is a decent shortcut to high accuracy — it just means they miss most of the actual high earners. 
    **We could've fixed this at training time with `class_weight='balanced'`** (tells the model to penalize mistakes on the rare class harder), but instead used the app's **threshold slider** — drag it below 0.50 to watch Recall go up (and Precision go down) live, no retraining needed.

- **Turning categories into plain numbers quietly hurts KNN and LR more than trees.** 
    Trees just ask yes/no questions, so it doesn't matter.
    KNN and LR do math on those numbers, so "category 4" ends up looking closer to "category 3" than it should — even though that's meaningless.

- **Tuning gave Decision Tree the biggest glow-up (AUC +0.136)** 
    Just by telling it to stop growing so deep. Turns out it wasn't a bad model, just an overconfident one.

- The **radar chart** (Compare All Models tab) makes this imbalance visible instead of just numeric
    LR and Naive Bayes visibly cave in on the Recall axis while Random Forest stays nice and round.

## Streamlit App Features
The deployed app ([link](https://2025ac05657-adult-income-classifier.streamlit.app)) allows the user to:
- Upload test data (CSV, raw/pre-encoding format, preferably the one in this repo for matching results) via the sidebar
- Select any of 8 models (5 baseline + 3 tuned variants of Decision Tree, KNN, Random Forest) from a dropdown
- View live-computed Accuracy, AUC, Precision, Recall, F1, and MCC for the selected model
- Adjust the decision threshold via an interactive slider to explore the precision/recall tradeoff without retraining
- View a confusion matrix and full classification report
- View an ROC curve with the AUC value annotated
- Switch to a "Compare All Models" tab to see all 8 models side-by-side in a condintionally-formatted table, a sortable bar chart per metric, and a radar chart for multi-metric shape comparison

*Note: metrics shown in the app are computed live on the uploaded CSV and will closely match (but may not be pixel-identical to) the notebook's reported results if a different-sized sample is uploaded.*

## Repository Structure
```
adult-income-classifier/
├── model/
│   └── train_models.ipynb
├── saved_models/
│   ├── decision_tree_tuned.pkl
│   ├── decision_tree.pkl
│   ├── knn_tuned.pkl
│   ├── knn.pkl
│   ├── label_encoders.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest_tuned.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── app.py
├── ML_Assignment_2.pdf
├── README.md
├── requirements.txt
└── test_data.csv
```

## How to Run Locally
```bash
git clone https://github.com/chocoblin/adult-income-classifier.git
cd adult-income-classifier
pip install -r requirements.txt
streamlit run app.py
```
Could be helpful to use a command like this in case streamlit command is not recognized by your terminal:

```
& "c:\Users\adityA\miniconda3\python.exe" -m streamlit run app.py
```

where the exact python path can be found out by doing the following in your venv of choice:

```
import sys
print(sys.executable)
```

---

Built with equal parts scikit-learn, google tabs, and mild panic about the 18-Aug deadline, the app went through more UI iterations than the actual model tuning. The confusion matrix alone survived a seaborn phase, a Plotly phase, and an "accidentally invisible white-on-white text" phase before it finally agreed to be sqyuare.
## If the Random Forest could talk, it would probably say "I told you so" to the other four models. And at this point, to the confusion matrix as well :D





