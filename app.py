import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page Configuration
st.set_page_config(
    page_title="Online Shoppers Intention Classifier",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Online Shoppers Purchasing Intention Classifier")
st.markdown(
    """
    This interactive application demonstrates 5 classification models trained to predict 
    whether an online session results in a transaction (**Revenue = True/False**).
    """
)

# Sidebar: Model and File Controls
st.sidebar.header("Configuration & Input")

# Model Mapping
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

selected_model_name = st.sidebar.selectbox(
    "Choose a Classification Model:",
    list(MODEL_FILES.keys())
)

# Rubric Requirement: CSV Upload Option
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"],
    help="Upload test_data.csv containing session features and the true 'Revenue' column."
)

use_default_data = st.sidebar.checkbox(
    "Or Use test_data.csv from repository",
    value=True if uploaded_file is None else False
)

# Data Loading Logic
df_test = None
if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom CSV uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded CSV: {e}")
elif use_default_data:
    if os.path.exists("test_data.csv"):
        df_test = pd.read_csv("test_data.csv")
        st.sidebar.info("Using repository default `test_data.csv`.")
    else:
        st.sidebar.warning("`test_data.csv` not found in root directory. Please upload a file.")

# Model Loading Function
@st.cache_resource
def load_trained_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Main Execution Flow
if df_test is not None:
    # Validate target column presence
    if 'Revenue' not in df_test.columns:
        st.error("Uploaded CSV must contain the ground-truth target column: 'Revenue'.")
    else:
        # Separate features and target
        X_test_input = df_test.drop(columns=['Revenue'])
        y_test_true = df_test['Revenue'].astype(int)

        # Load selected model
        model_path = MODEL_FILES[selected_model_name]
        model = load_trained_model(model_path)

        if model is None:
            st.error(f"Model file not found at `{model_path}`. Please ensure training artifacts exist.")
        else:
            # Predict labels and probability
            y_pred = model.predict(X_test_input)
            
            # Predict probability if supported
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_input)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test_input)
            else:
                y_proba = y_pred

            # Compute Evaluation Metrics
            acc = accuracy_score(y_test_true, y_pred)
            auc = roc_auc_score(y_test_true, y_proba)
            prec = precision_score(y_test_true, y_pred, zero_division=0)
            rec = recall_score(y_test_true, y_pred, zero_division=0)
            f1 = f1_score(y_test_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test_true, y_pred)

            # Display Metrics
            st.subheader(f"📊 Performance Metrics: {selected_model_name}")
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("Accuracy", f"{acc * 100:.2f}%")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")

            # Diagnostics: Confusion Matrix & Classification Report
            col_diag1, col_diag2 = st.columns([1, 1])

            with col_diag1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test_true, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["No Purchase (0)", "Purchase (1)"],
                    yticklabels=["No Purchase (0)", "Purchase (1)"],
                    cbar=False,
                    ax=ax
                )
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

            with col_diag2:
                st.subheader("Classification Report")
                report_dict = classification_report(
                    y_test_true,
                    y_pred,
                    target_names=["No Purchase (0)", "Purchase (1)"],
                    output_dict=True
                )
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)

            st.markdown("---")

            # Test Dataset Sample Viewer
            with st.expander("🔍 View Uploaded Test Data Sample (First 5 Rows)"):
                st.dataframe(df_test.head())

else:
    st.info("👈 Upload a test dataset (`test_data.csv`) using the sidebar to begin model evaluation.")