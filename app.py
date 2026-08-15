import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                               recall_score, f1_score, matthews_corrcoef,
                               confusion_matrix, classification_report)

st.set_page_config(page_title="Adult Income Classifier", layout="wide")

st.title("Adult Income Classification — Model Comparison")
st.write(
    "Upload test data and compare 5 classification models (baseline + tuned variants) trained to predict whether an individual earns more than $50K/year based on census attributes."
)

MODEL_FILES = {
    "Logistic Regression": "saved_models/logistic_regression.pkl",
    "Decision Tree (baseline)": "saved_models/decision_tree.pkl",
    "Decision Tree (tuned)": "saved_models/decision_tree_tuned.pkl",
    "KNN (baseline)": "saved_models/knn.pkl",
    "KNN (tuned)": "saved_models/knn_tuned.pkl",
    "Naive Bayes": "saved_models/naive_bayes.pkl",
    "Random Forest (baseline)": "saved_models/random_forest.pkl",
    "Random Forest (tuned)": "saved_models/random_forest_tuned.pkl",
}

SCALER_FILE = "saved_models/scaler.pkl"
ENCODERS_FILE = "saved_models/label_encoders.pkl"

# 
@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.sidebar.header("Setup")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info("Upload a CSV file using the sidebar to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Uploaded Data Preview (raw)")
st.dataframe(df.head())

if "income" not in df.columns:
    st.error("The uploaded CSV must contain an 'income' column (the true label) "
              "to evaluate model performance.")
    st.stop()

y_true = df["income"].values
X_df = df.drop("income", axis=1).copy()

# Drop education_num — dropped during training as a duplicate of 'education'
if "education_num" in X_df.columns:
    X_df = X_df.drop("education_num", axis=1)

# Apply the SAME LabelEncoders fitted during training to each categorical column
try:
    encoders = load_model(ENCODERS_FILE)
    for col, encoder in encoders.items():
        if col in X_df.columns:
            unseen = set(X_df[col].astype(str).str.strip()) - set(encoder.classes_)
            if unseen:
                st.error(f"Column '{col}' contains categories not seen during "
                          f"training: {unseen}. Cannot encode this data.")
                st.stop()
            X_df[col] = encoder.transform(X_df[col].astype(str).str.strip())
except Exception as e:
    st.error(f"Error encoding categorical columns: {e}")
    st.stop()

X = X_df.values

try:
    scaler = load_model(SCALER_FILE)
    X_scaled = scaler.transform(X)
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()

try:
    model = load_model(MODEL_FILES[selected_model_name])
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_proba),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.subheader(f"Evaluation Metrics — {selected_model_name}")
cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred,
                                     target_names=["<=50K", ">50K"],
                                     output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

st.caption("Model, scaler, and preprocessing pipeline trained on the UCI "
           "Adult Income dataset. See README for full methodology and "
           "model comparison.")