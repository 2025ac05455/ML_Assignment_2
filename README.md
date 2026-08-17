# Machine Learning Assignment 2: Classification Model Evaluation Dashboard

**Course**: M.Tech (AIML / DSE) – BITS Pilani WILP  
**GitHub Repository**: [https://github.com/2025ac05455/ML_Assignment_2](https://github.com/2025ac05455/ML_Assignment_2)  

---

## a. Problem Statement
The objective of this assignment is to build, evaluate, and compare multiple supervised machine learning classification algorithms on a tabular chemical properties dataset to accurately classify wine types (Red vs. White). The project evaluates five classification models across six standard performance metrics (Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient) and deploys an interactive web interface using Streamlit Community Cloud for dynamic evaluation and model comparison.

---

## b. Dataset Description
* **Source**: UCI Machine Learning Repository (Wine Quality Dataset)
* **Total Instances**: 6,497 samples (Exceeds the minimum threshold of 500 instances)
* **Total Features**: 12 continuous numerical attributes (Exceeds the minimum requirement of 12 features)
* **Attribute Features**:
  1. `fixed_acidity`
  2. `volatile_acidity`
  3. `citric_acid`
  4. `residual_sugar`
  5. `chlorides`
  6. `free_sulfur_dioxide`
  7. `total_sulfur_dioxide`
  8. `density`
  9. `pH`
  10. `sulphates`
  11. `alcohol`
  12. `quality`
* **Target Variable**: `target` (Binary Classification: `1` = Red Wine, `0` = White Wine)

---

## c. GitHub Repository Link
[https://github.com/2025ac05455/ML_Assignment_2](https://github.com/2025ac05455/ML_Assignment_2)

---

## d. Models Used & Comparison Tables

### 1. Performance Metric Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.996154 | 0.994869 | 0.993730 | 0.990625 | 0.992175 | 0.989628 |
| **Decision Tree** | 0.989231 | 0.984518 | 0.978125 | 0.978125 | 0.978125 | 0.970982 |
| **kNN** | 0.993077 | 0.997758 | 0.984424 | 0.987500 | 0.985959 | 0.981367 |
| **Naive Bayes** | 0.980769 | 0.985405 | 0.951070 | 0.971875 | 0.961360 | 0.948660 |
| **Random Forest (Ensemble)** | **0.998462** | **1.000000** | **1.000000** | **0.993750** | **0.996865** | **0.995854** |

*(Exact unrounded values matching terminal execution output).*

---

### 2. Model Performance Observations Table

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved an impressive **0.996154 Accuracy** and **0.989628 MCC**. This confirms strong linear separability between red and white wines based on chemical parameters like total sulfur dioxide and volatile acidity. |
| **Decision Tree** | Reached a high accuracy of **0.989231**, though it exhibited slightly lower overall generalization (AUC **0.984518**) compared to ensemble methods due to standard axis-aligned decision boundary splits. |
| **kNN** | Delivered exceptional results (**0.993077 Accuracy**, **0.997758 AUC**) once feature standardization was applied, efficiently capturing local neighborhood clusters in the chemical feature space. |
| **Naive Bayes** | Provided fast training with strong recall (**0.971875**), though feature correlation assumption violations slightly lowered its precision (**0.951070**) relative to other models. |
| **Random Forest (Ensemble)** | **Top Overall Performer**: Outperformed every other model across all metrics with a near-perfect **0.998462 Accuracy**, **1.000000 AUC**, and **0.995854 MCC** by aggregating multiple decision trees to effectively eliminate model variance. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** is the undisputed top-performing model across all 6 evaluation metrics. |

---

## 🛠️ Repository Structure
```text
ML_Assignment_2/
├── model/
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── random_forest.pkl
├── app.py
├── model_training.py
├── README.md
├── requirements.txt
└── test_data.csv
