import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                               recall_score, f1_score, matthews_corrcoef,
                               confusion_matrix, classification_report, roc_curve)

st.set_page_config(page_title="Adult Income Classifier", layout="wide")

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


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def preprocess(df, encoders):
    """Raw uploaded CSV -> encoded, scaler-ready X and y_true."""
    y_true = df["income"].values
    X_df = df.drop("income", axis=1).copy()

    if "education_num" in X_df.columns:
        X_df = X_df.drop("education_num", axis=1)

    for col, encoder in encoders.items():
        if col in X_df.columns:
            unseen = set(X_df[col].astype(str).str.strip()) - set(encoder.classes_)
            if unseen:
                raise ValueError(
                    f"Column '{col}' has categories not seen during training: {unseen}"
                )
            X_df[col] = encoder.transform(X_df[col].astype(str).str.strip())

    return X_df.values, y_true


def compute_metrics(model, X_scaled, y_true):
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }, y_pred


# ---------------- Sidebar ----------------
st.sidebar.markdown("**Aditya**  \nGitHub: [chocoblin](https://github.com/chocoblin)  \nBITS Roll No: 2025AC05657")
st.sidebar.divider()
st.sidebar.header("Setup")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Persist the uploaded file's content in session_state so switching tabs /
    # dropdowns elsewhere doesn't require re-uploading (Streamlit reruns the
    # whole script on every interaction, wiping local variables each time).
    st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)

if "uploaded_df" not in st.session_state:
    st.title("Adult Income Classification — Model Comparison")
    st.write(
        "Upload test data to compare 5 classification models (baseline and "
        "tuned variants) trained to predict whether an individual earns more "
        "than $50K/year based on census attributes."
    )
    st.info("Upload a CSV file using the sidebar to get started.")
    st.stop()

df = st.session_state["uploaded_df"]

if "income" not in df.columns:
    st.error("The uploaded CSV must contain an 'income' column (the true label) "
              "to evaluate model performance.")
    st.stop()

try:
    encoders = load_model(ENCODERS_FILE)
    scaler = load_model(SCALER_FILE)
    X, y_true = preprocess(df, encoders)
    X_scaled = scaler.transform(X)
except Exception as e:
    st.error(f"Error preprocessing uploaded data: {e}")
    st.stop()

st.title("Adult Income Classification — Model Comparison")

with st.expander("Uploaded Data Preview", expanded=False):
    st.dataframe(df.head())
    st.caption(f"{df.shape[0]} rows, {df.shape[1]} columns")

tab1, tab2 = st.tabs(["Single Model", "Compare All Models"])

# ---------------- Tab 1: single model deep dive ----------------
with tab1:
    selected_model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))

    try:
        model = load_model(MODEL_FILES[selected_model_name])
        metrics, y_pred = compute_metrics(model, X_scaled, y_true)
        y_proba = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        st.error(f"Error running model: {e}")
        st.stop()

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

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=selected_model_name,
        line=dict(color="#3366CC", width=3),
        fill="tozeroy", fillcolor="rgba(51,102,204,0.15)",
    ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
        line=dict(color="gray", width=1, dash="dash"),
    ))
    roc_fig.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=420, legend=dict(x=0.6, y=0.1),
        annotations=[dict(x=0.55, y=0.05, xref="paper", yref="paper",
                           text=f"AUC = {metrics['AUC']:.4f}", showarrow=False,
                           font=dict(size=14))],
    )
    st.plotly_chart(roc_fig, use_container_width=True)

# ---------------- Tab 2: all models compared ----------------
with tab2:
    st.subheader("All Models — Side by Side")

    with st.spinner("Running all 8 models on your uploaded data..."):
        all_results = {}
        for name, path in MODEL_FILES.items():
            m = load_model(path)
            metrics, _ = compute_metrics(m, X_scaled, y_true)
            all_results[name] = metrics

    results_df = pd.DataFrame(all_results).T
    results_df.index.name = "Model"

    st.dataframe(
        results_df.style.format("{:.4f}").background_gradient(
            cmap="Greens", axis=0
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Metric Comparison")
        metric_to_plot = st.selectbox(
            "Metric to compare", results_df.columns, index=5
        )
        plot_df = results_df[[metric_to_plot]].reset_index().sort_values(
            metric_to_plot, ascending=True
        )
        bar_fig = px.bar(
            plot_df, x=metric_to_plot, y="Model", orientation="h",
            color=metric_to_plot, color_continuous_scale="Blues",
            text_auto=".3f",
        )
        bar_fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(bar_fig, use_container_width=True)

    with col2:
        st.subheader("Overall Shape (Radar)")
        radar_models = st.multiselect(
            "Models to include", results_df.index.tolist(),
            default=[results_df["MCC"].idxmax(), results_df["MCC"].idxmin()],
        )
        if radar_models:
            radar_fig = go.Figure()
            categories = list(results_df.columns)
            for name in radar_models:
                values = results_df.loc[name, categories].tolist()
                radar_fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself", name=name, opacity=0.6,
                ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=420, showlegend=True,
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Select at least one model to display the radar chart.")

    best_model = results_df["MCC"].idxmax()
    best_on_selected = results_df[metric_to_plot].idxmax()
    st.success(
        f"Best on {metric_to_plot}: {best_on_selected} "
        f"({results_df.loc[best_on_selected, metric_to_plot]:.4f})  |  "
        f"Best overall (MCC): {best_model} ({results_df.loc[best_model, 'MCC']:.4f})"
    )

st.caption("Model, scaler, and preprocessing pipeline trained on the UCI "
           "Adult Income dataset. Metrics are computed live on your uploaded "
           "sample and may differ slightly from the full-test-set results "
           "reported in the training notebook/README due to sample size. "
           "See README for full methodology and model comparison.")