import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Page configuration
st.set_page_config(
    page_title="Airline Passenger Satisfaction Classifier",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Airline Passenger Satisfaction Prediction Dashboard")
st.markdown(
    """
This interactive web application evaluates five distinct classification models on test data without retraining, 
presenting multi-metric performance evaluations, confusion matrices, and detailed classification reports.
"""
)

# Sidebar: Controls
st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must include ground truth 'satisfaction')", type=["csv"]
)

st.sidebar.header("2. Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Classifier",
    [
        "Support Vector Machine",
        "Decision Tree",
        "KNN",
        "Logistic Regression",
        "Naive Bayes",
    ],
)

# Model file mapper
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def load_artifacts():
    prep = joblib.load("model/preprocessor.joblib")
    med = joblib.load("model/arrival_median.joblib")
    return prep, med


if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    if os.path.exists("test_data.csv"):
        test_df = pd.read_csv("test_data.csv")
        st.info("Loaded default repository test dataset (`test_data.csv`).")
    else:
        st.warning("Please upload a test dataset CSV to proceed.")
        st.stop()

# Data Preview
with st.expander("🔍 Preview Test Data", expanded=False):
    st.dataframe(test_df.head(10))

# Verification of Target
if "satisfaction" not in test_df.columns:
    st.error(
        "The uploaded CSV must contain the target column labeled 'satisfaction'."
    )
    st.stop()

# Execution
try:
    preprocessor, arrival_median = load_artifacts()

    # Preprocessing
    X_test = test_df.drop(columns=["satisfaction"]).copy()
    if "Arrival Delay in Minutes" in X_test.columns:
        X_test["Arrival Delay in Minutes"] = X_test[
            "Arrival Delay in Minutes"
        ].fillna(arrival_median)

    y_test = test_df["satisfaction"].map(
        {"neutral or dissatisfied": 0, "satisfied": 1}
    )
    if y_test.isnull().any():
        # Handle 0/1 numeric targets if already mapped
        y_test = test_df["satisfaction"].astype(int)

    X_test_proc = preprocessor.transform(X_test)

    # Load Model
    model_path = model_files[model_choice]
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X_test_proc)
    y_proba = (
        model.predict_proba(X_test_proc)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

    st.subheader(f"📊 Performance Metrics: {model_choice}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("AUC Score", f"{auc:.4f}")
    m3.metric("Precision", f"{prec:.4f}")
    m4.metric("Recall", f"{rec:.4f}")
    m5.metric("F1 Score", f"{f1:.4f}")
    m6.metric("MCC Score", f"{mcc:.4f}")

    st.markdown("---")

    col_cm, col_cr = st.columns([1, 1])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Dissatisfied (0)", "Satisfied (1)"],
            yticklabels=["Dissatisfied (0)", "Satisfied (1)"],
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

    with col_cr:
        st.subheader("Classification Report")
        cr_dict = classification_report(
            y_test,
            y_pred,
            target_names=["Neutral/Dissatisfied", "Satisfied"],
            output_dict=True,
        )
        cr_df = pd.DataFrame(cr_dict).transpose()
        st.dataframe(cr_df.style.format(precision=4))

except Exception as e:
    st.error(f"Error during prediction: {e}")