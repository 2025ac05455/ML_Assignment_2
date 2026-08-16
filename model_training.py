import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Create output folder for model artifacts
os.makedirs("model", exist_ok=True)

# Load UCI Wine Quality Dataset (Red + White combined: 6,497 rows, 12 features)
red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_df = pd.read_csv(red_url, sep=';')
white_df = pd.read_csv(white_url, sep=';')

# Binary target: 1 for Red Wine, 0 for White Wine
red_df['target'] = 1
white_df['target'] = 0

df = pd.concat([red_df, white_df], ignore_index=True)

X = df.drop(columns=['target'])
y = df['target']

# 80/20 Train-Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Export unscaled test data CSV with 'target' column for Streamlit upload
test_export = X_test.copy()
test_export['target'] = y_test
test_export.to_csv("test_data.csv", index=False)

# Define pipeline preprocessor
scaler = StandardScaler()

# Define exact 5 classification models
models = {
    "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree": Pipeline([('clf', DecisionTreeClassifier(max_depth=10, random_state=42))]),
    "kNN": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
    "Random Forest (Ensemble)": Pipeline([('clf', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))])
}

results = []

# Train and compute all 6 required metrics
for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Save trained pipeline model
    file_key = name.lower().replace(" ", "_").replace("_(ensemble)", "")
    joblib.dump(model, f"model/{file_key}.pkl")
    
    results.append({
        "Model": name, "Accuracy": acc, "AUC": auc,
        "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc
    })

# Print evaluation table to terminal
metrics_df = pd.DataFrame(results)
print("\n================ Model Evaluation Summary ================")
print(metrics_df.to_string(index=False))