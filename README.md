# Machine Learning Assignment 2: Classification Model Evaluation Dashboard

## a. Problem Statement
The goal of this project is to implement and compare multiple machine learning classification algorithms on a tabular chemical properties dataset to predict wine type (Red vs. White). The models are evaluated on six standard performance metrics, and the entire workflow is deployed as an interactive web application on Streamlit Community Cloud.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository (Wine Quality Dataset)
- **Instances**: 6,497 total samples (1,599 Red Wine, 4,898 White Wine)
- **Features**: 12 continuous numerical features (`fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality`)
- **Target Column**: `target` (Binary: 1 = Red Wine, 0 = White Wine)

## c. Github Repository Link
`https://github.com/<YOUR_GITHUB_USERNAME>/ML_Assignment_2`

## d. Models Used

### Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9885 | 0.9982 | 0.9810 | 0.9719 | 0.9764 | 0.9691 |
| **Decision Tree** | 0.9838 | 0.9856 | 0.9686 | 0.9656 | 0.9671 | 0.9568 |
| **kNN** | 0.9923 | 0.9971 | 0.9905 | 0.9750 | 0.9827 | 0.9774 |
| **Naive Bayes** | 0.9731 | 0.9891 | 0.9169 | 0.9781 | 0.9465 | 0.9312 |
| **Random Forest (Ensemble)** | **0.9954** | **0.9996** | **0.9937** | **0.9875** | **0.9906** | **0.9877** |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well due to distinct linear separation in chemical composition between red and white wines. |
| **Decision Tree** | Achieved high baseline accuracy but slightly lower generalization compared to ensemble methods due to Axis-aligned splits. |
| **kNN** | Highly effective once features were standardized, capturing local clusters in feature space accurately. |
| **Naive Bayes** | Delivered fast training and strong recall, though precision was lower due to slight correlations among chemical features. |
| **Random Forest (Ensemble)** | Outperformed all other models across every metric by aggregating multiple decision trees to minimize prediction variance. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** is the top-performing model with an Accuracy of **0.9954**, AUC of **0.9996**, and MCC of **0.9877**. |
