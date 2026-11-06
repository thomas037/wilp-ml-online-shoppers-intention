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
    classification_report,
    roc_curve
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Online Shoppers Intention Classifier",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar: Branding, Model Selection & File Upload
# ---------------------------------------------------------
with st.sidebar:
    # Sidebar local image banner
    if os.path.exists("banner.gif"):
        st.image("banner.gif", use_container_width=True)
    
    st.header("⚙️ Configuration & Input")
    
    # 5 Models mapped to joblib pipeline artifacts
    MODEL_FILES = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "kNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "Support Vector Machine": "model/support_vector_machine.joblib"
    }

    selected_model_name = st.selectbox(
        "Choose a Classification Model:",
        list(MODEL_FILES.keys())
    )

    # Mandatory Rubric Item: CSV Dataset Upload
    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
        help="Upload test_data.csv containing session features and the true 'Revenue' column."
    )

    use_default_data = st.checkbox(
        "Use repository default test_data.csv",
        value=True if uploaded_file is None else False
    )

    st.markdown("---")

# ---------------------------------------------------------
# Data Loading & Pipeline Verification
# ---------------------------------------------------------
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

@st.cache_resource
def load_trained_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.title("🛒 Online Shoppers Purchasing Intention Classifier")
st.markdown(
    "Interactive Machine Learning deployment benchmarking classification models "
    "on the **UCI Online Shoppers Purchasing Intention** dataset."
)

if df_test is not None:
    # Ensure ground-truth target is present
    if 'Revenue' not in df_test.columns:
        st.error("Uploaded CSV must contain the ground-truth target column: 'Revenue'.")
    else:
        # Guarantee engineered features exist if raw data was uploaded
        if 'Avg_Duration_Per_Page' not in df_test.columns or 'Product_Ratio' not in df_test.columns:
            total_pages = df_test['Administrative'] + df_test['Informational'] + df_test['ProductRelated']
            total_duration = (
                df_test['Administrative_Duration'] 
                + df_test['Informational_Duration'] 
                + df_test['ProductRelated_Duration']
            )
            df_test['Avg_Duration_Per_Page'] = total_duration / (total_pages + 1e-5)
            df_test['Product_Ratio'] = df_test['ProductRelated'] / (total_pages + 1e-5)

        X_test_input = df_test.drop(columns=['Revenue'])
        y_test_true = df_test['Revenue'].astype(int)

        # Load active model
        model_path = MODEL_FILES[selected_model_name]
        model = load_trained_model(model_path)

        if model is None:
            st.error(f"Model file not found at `{model_path}`. Please run training to export artifacts.")
        else:
            # Predict labels & probabilities
            y_pred = model.predict(X_test_input)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_input)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test_input)
            else:
                y_proba = y_pred

            # Calculate the 6 required metrics
            acc = accuracy_score(y_test_true, y_pred)
            auc = roc_auc_score(y_test_true, y_proba)
            prec = precision_score(y_test_true, y_pred, zero_division=0)
            rec = recall_score(y_test_true, y_pred, zero_division=0)
            f1 = f1_score(y_test_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test_true, y_pred)

            # Sidebar download button for predicted results
            results_export = df_test.copy()
            results_export["Predicted_Revenue"] = y_pred
            results_export["Purchase_Probability"] = np.round(y_proba, 4)

            csv_export_bytes = results_export.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="📥 Download Predictions CSV",
                data=csv_export_bytes,
                file_name=f"{selected_model_name.lower().replace(' ', '_')}_predictions.csv",
                mime="text/csv",
                help="Download test data records with model predictions and probabilities"
            )

            # UI Customization: Tabbed Organization
            tab_eval, tab_leaderboard, tab_data = st.tabs([
                "🎯 Model Evaluation",
                "📈 All Models Leaderboard",
                "📁 Dataset Explorer"
            ])

            # -------------------------------------------------
            # TAB 1: MODEL EVALUATION
            # -------------------------------------------------
            with tab_eval:
                st.subheader(f"Performance Scorecard: {selected_model_name}")

                # Metric Cards
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Accuracy", f"{acc * 100:.2f}%")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")

                # Visual callout summary
                purchase_pct = (y_pred.sum() / len(y_pred)) * 100
                st.info(
                    f"💡 **Prediction Breakdown:** Classified **{y_pred.sum():,}** out of "
                    f"**{len(y_pred):,}** sessions (**{purchase_pct:.1f}%**) as completed transactions."
                )

                st.markdown("---")

                # Visual Diagnostics: Confusion Matrix & ROC Curve
                col_cm, col_roc = st.columns(2)

                with col_cm:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test_true, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["No Purchase (0)", "Purchase (1)"],
                        yticklabels=["No Purchase (0)", "Purchase (1)"],
                        cbar=False,
                        ax=ax_cm
                    )
                    ax_cm.set_xlabel("Predicted Label")
                    ax_cm.set_ylabel("True Label")
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                with col_roc:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test_true, y_proba)
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 3.8))
                    ax_roc.plot(fpr, tpr, color='#1E3C72', lw=2.5, label=f'ROC (AUC = {auc:.4f})')
                    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.05])
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.legend(loc="lower right")
                    ax_roc.grid(alpha=0.3)
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)

                st.markdown("---")

                # Classification Report Table
                st.subheader("Classification Report")
                report_dict = classification_report(
                    y_test_true,
                    y_pred,
                    target_names=["No Purchase (0)", "Purchase (1)"],
                    output_dict=True
                )
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)

            # -------------------------------------------------
            # TAB 2: ALL MODELS LEADERBOARD
            # -------------------------------------------------
            with tab_leaderboard:
                st.subheader("Comparative Benchmark Across All 5 Models")
                st.markdown("Evaluation metrics dynamically computed across all trained models on the active test set:")

                leaderboard_rows = []
                for name, path in MODEL_FILES.items():
                    m = load_trained_model(path)
                    if m is not None:
                        pred = m.predict(X_test_input)
                        if hasattr(m, "predict_proba"):
                            proba = m.predict_proba(X_test_input)[:, 1]
                        elif hasattr(m, "decision_function"):
                            proba = m.decision_function(X_test_input)
                        else:
                            proba = pred

                        leaderboard_rows.append({
                            "ML Model Name": name,
                            "Accuracy": round(accuracy_score(y_test_true, pred), 4),
                            "AUC": round(roc_auc_score(y_test_true, proba), 4),
                            "Precision": round(precision_score(y_test_true, pred, zero_division=0), 4),
                            "Recall": round(recall_score(y_test_true, pred, zero_division=0), 4),
                            "F1": round(f1_score(y_test_true, pred, zero_division=0), 4),
                            "MCC": round(matthews_corrcoef(y_test_true, pred), 4)
                        })

                if leaderboard_rows:
                    bench_df = pd.DataFrame(leaderboard_rows)
                    st.dataframe(
                        bench_df.style.highlight_max(
                            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], 
                            color="#d1e7dd"
                        ),
                        use_container_width=True
                    )
                else:
                    st.warning("Ensure all 5 model artifacts exist in the `model/` folder to populate the leaderboard.")

            # -------------------------------------------------
            # TAB 3: DATASET EXPLORER
            # -------------------------------------------------
            with tab_data:
                st.subheader("Active Test Dataset Explorer")
                st.write(f"Total Rows: **{df_test.shape[0]}** | Total Columns: **{df_test.shape[1]}**")
                st.dataframe(df_test, use_container_width=True)
                
                with st.expander("📊 View Numerical Summary Statistics"):
                    st.dataframe(df_test.describe().T, use_container_width=True)

else:
    st.info("👈 Upload a test dataset (`test_data.csv`) using the sidebar to begin model evaluation.")

# ---------------------------------------------------------
# Sidebar Footer: Course & Submission Metadata
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        **BITS Pilani — WILP**  
        *Machine Learning Assignment 2*  
        
        **Author:** Leo Thomas
        
        **Dataset:** UCI Online Shoppers (12,330 records)
        """
    )