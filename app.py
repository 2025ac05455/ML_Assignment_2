import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Wine Classification Evaluator", layout="wide")

st.title("🍷 Machine Learning Model Performance Dashboard")
st.caption("BITS Pilani WILP - Machine Learning Assignment 2")

st.sidebar.header("⚙️ Dashboard Controls")

# Step 6a: Dataset upload option
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# Step 6b: Model selection dropdown
model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest (Ensemble)")
)

model_path_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl"
}

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head(5))
    
    if "target" in data.columns:
        X_test = data.drop(columns=["target"])
        y_test = data["target"]
        
        # Load pipeline and predict
        model = joblib.load(model_path_map[model_choice])
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Step 6c: Display evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.subheader(f"📊 Metrics for **{model_choice}**")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.divider()
        
        # Step 6d: Confusion matrix & classification report
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with c_right:
            st.subheader("📝 Classification Report")
            rep = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(rep).transpose())
            
    else:
        st.error("Uploaded CSV must contain a 'target' column.")
else:
    st.info("👈 Please upload `test_data.csv` using the sidebar menu to view results.")