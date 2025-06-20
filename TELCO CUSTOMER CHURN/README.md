# Telco Customer Churn Prediction using Logistic Regression & Random Forest

## Project Overview

This project aims to predict customer churn for a telecom company by leveraging machine learning models on a dataset containing customer demographics, service details, and billing information. Accurate churn prediction helps the company proactively retain customers and reduce revenue loss.

## Data and Feature Engineering

- Selected **20 key features** based on Random Forest feature importance, including both raw and engineered features such as `TotalCharges`, `tenure`, `MonthlyCharges`, contract type, and various service flags.
- Features were split into numerical and categorical groups for appropriate preprocessing:
  - Numerical features standardized using `StandardScaler`.
  - Categorical features encoded using `OneHotEncoder` with `handle_unknown='ignore'` to accurately handle unseen categories.
- Data split into training and testing sets using stratified sampling to preserve churn class distribution.

## Modeling Approaches

### Random Forest Classifier

- Initial baseline Random Forest trained with default parameters and balanced class weights to address churn imbalance.
- Hyperparameter tuning performed using `RandomizedSearchCV` over parameters such as number of estimators, max depth, max features, and minimum samples for split and leaf nodes.
- The best model achieved:
  - **ROC AUC Score:** ~0.84
  - Improved recall on the churn class (77%), crucial for minimizing false negatives (missed churners).
  - Balanced precision and recall leading to a more reliable classification performance.
- Confusion matrix and classification report demonstrate a strong ability to identify both churners and non-churners effectively.

### Logistic Regression Model

- Developed a Logistic Regression baseline model to provide a simpler, interpretable comparison.
- Features underwent the same preprocessing pipeline as Random Forest.
- Logistic Regression used with class weights balanced to handle class imbalance.
- Model evaluation showed:
  - Slightly lower ROC AUC '0.839' compared to Random Forest.
  - Precision and recall trade-offs consistent with its linear nature.
- Logistic Regression offers valuable insights into linear feature relationships with churn probability, acting as a strong benchmark.

## Model Evaluation

- Both models evaluated on unseen test data.
- Metrics used:
  - **Precision, Recall, F1-Score** per class.
  - **Confusion Matrix** to understand types of classification errors.
  - **ROC AUC** to measure overall discrimination ability.
- Random Forest outperformed Logistic Regression in capturing nonlinear feature interactions, thus providing better recall for the churn class.

## Next Steps

- Deploy the tuned Random Forest model for real-time churn prediction in production.
- Integrate SHAP explanations into customer retention dashboards to explain individual predictions.
- Explore additional modeling techniques (e.g., Gradient Boosting, Neural Networks) and ensemble methods.
- Investigate data enrichment and longitudinal customer behavior for improved prediction accuracy.

---

## Usage Instructions

- The notebook contains the entire workflow: data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and interpretability steps.
- Load the dataset and execute notebook cells sequentially.
- Modify hyperparameters and features as needed for further experimentation.
