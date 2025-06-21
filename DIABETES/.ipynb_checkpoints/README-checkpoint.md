# Diabetes Prediction Project

## Overview

This project is an end-to-end data science pipeline to predict diabetes using the Pima Indians Diabetes Dataset. It showcases proficiency in data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning modeling, and evaluation. The goal is to accurately identify diabetic patients, prioritizing recall to minimize missed diagnoses, which is critical in medical applications. The best model, a tuned Random Forest with a classification threshold of 0.4, achieves 83.6% recall and 0.831 ROC-AUC.

## Dataset

**Source:** Pima Indians Diabetes Database (Kaggle)

**Description:** 768 records with 9 features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = non-diabetic, 1 = diabetic)

**Challenges:**

- Unrealistic zeros (e.g., 48.7% of Insulin, 29.6% of SkinThickness)
- Class imbalance (65% non-diabetic, 35% diabetic)
- Outliers (e.g., Insulin max=846)

## Project Structure
Diabetes-Prediction-Project/
├── data/
│ ├── diabetes.csv # Original dataset
│ └── processed_diabetes.csv # Processed dataset
├── notebooks/
│ ├── 01_data_loading.ipynb # Load and inspect data
│ ├── 02_eda.ipynb # Exploratory data analysis
│ ├── 03_feature_engineering.ipynb # Data preprocessing
│ └── 04_modeling.ipynb # Model training and evaluation
├── models/
│ └── best_rf_model.pkl # Best Random Forest model
└── README.md

## Requirements

- Python 3.8+
- Libraries:  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`

## Methodology

### 1. Data Loading (`01_data_loading.ipynb`)

- Loaded dataset using Pandas.
- Inspected structure (768 rows, 9 columns).
- Identified zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI.

### 2. Exploratory Data Analysis (`02_eda.ipynb`)

- **Summary Statistics:** Revealed zeros (e.g., 48.7% in Insulin) and outliers (e.g., Insulin max=846).
- **Class Distribution:** 65% non-diabetic, 35% diabetic (imbalanced).
- **Visualizations:**
  - Histograms: Showed skewed distributions (e.g., Insulin).
  - Boxplots: Highlighted outliers.
  - Correlation Heatmap: Glucose-Outcome correlation = 0.47.
  - Zero Analysis: High zero proportions in Insulin (48.7%) and SkinThickness (29.6%).

### 3. Feature Engineering (`03_feature_engineering.ipynb`)

- **Zero Handling:** Replaced zeros with median values.
- **Outlier Treatment:** Capped outliers using IQR method (e.g., Insulin max reduced to 135.875).
- **New Features:**
  - `BMI_Category`: Categorized into Underweight, Normal, Overweight, Obese.
  - `Glucose_Insulin_Ratio`: Captured interaction between Glucose and Insulin.
- **Scaling:** Standardized numeric features; one-hot encoded `BMI_Category`.
- **Output:** Saved processed dataset (`processed_diabetes.csv`).

### 4. Modeling (`04_modeling.ipynb`)

- **Train-Test Split:** 80% train (614 rows), 20% test (154 rows).
- **Class Imbalance:** Applied SMOTE to balance training data (802 rows, 50:50).

---

### Models Trained

#### Logistic Regression

- **Description:** A linear model suitable for binary classification, interpretable, and robust to feature scaling.
- **Hyperparameters:** Default settings with `max_iter=1000` for convergence.
- **Performance (test set):**
  - Accuracy: 0.721
  - Precision: 0.583
  - Recall: 0.764
  - F1-Score: 0.661
  - ROC-AUC: 0.825
- **Confusion Matrix:**
[[69 30]] # TN=69, FP=30
[[13 42]] # FN=13, TP=42
- **Insights:**
- Highest recall (0.764) among default models, crucial for minimizing missed diabetic cases.
- Lower precision (0.583) indicates more false positives, a trade-off for high recall.
- Strong ROC-AUC (0.825) shows good discrimination between classes.
- Suitable for baseline modeling but limited by linear assumptions.

#### Random Forest

- **Description:** An ensemble of decision trees, robust to overfitting and effective for non-linear relationships.
- **Initial Performance (default settings):**
- Accuracy: 0.753
- Precision: 0.639
- Recall: 0.709
- F1-Score: 0.672
- ROC-AUC: 0.831
- **Confusion Matrix:**
[[77 22]] # TN=77, FP=22
[[16 39]] # FN=16, TP=39
- **Tuned Hyperparameters:** Used GridSearchCV to optimize:
- `max_depth=10`, `n_estimators=200`, `min_samples_split=2`
- F1-Score improved to 0.689
- **Threshold Adjustment:** Set classification threshold to 0.4 to prioritize recall.
- **Performance (threshold=0.4):**
- Recall: 0.836
- F1-Score: 0.687
- **Confusion Matrix:**

- **Feature Importance:**
- Top features: Glucose (0.197), Glucose_Insulin_Ratio (0.151), Age (0.142).
- Validates the engineered feature (`Glucose_Insulin_Ratio`) and biological relevance of Glucose.
- **Insights:**
- Best overall model (highest ROC-AUC: 0.831, balanced metrics).
- Threshold adjustment increased recall to 0.836, reducing false negatives to 9, ideal for medical use.
- Robust to non-linear patterns and feature interactions.
- **Saved as** `best_rf_model.pkl`.

#### XGBoost

- **Description:** A gradient boosting algorithm optimized for speed and performance, effective for imbalanced datasets.
- **Hyperparameters:** Default settings with `eval_metric='logloss'`.
- **Performance (test set):**
- Accuracy: 0.740
- Precision: 0.612
- Recall: 0.745
- F1-Score: 0.672
- ROC-AUC: 0.803
- **Confusion Matrix:**
[[73 26]] # TN=73, FP=26
[[14 41]] # FN=14, TP=41
- **Insights:**
- Competitive recall (0.745) and F1-Score (0.672), but lower ROC-AUC (0.803) than Random Forest.
- Strong performance on imbalanced data due to boosting, but less effective than Random Forest here.
- Potential for improvement with hyperparameter tuning (e.g., learning_rate, max_depth).

---

### Visualizations

- Confusion matrices for each model to show true/false positives/negatives.
- ROC curves comparing model discrimination (Random Forest AUC=0.831 highest).
- Feature importance plot for Random Forest, highlighting Glucose and Glucose_Insulin_Ratio.

---

### Output

- Saved best Random Forest model as `best_rf_model.pkl`.

## Results

**Best Model:** Random Forest (threshold=0.4)

- **Recall:** 0.836 (identifies 83.6% of diabetics)  
- **F1-Score:** 0.687  
- **ROC-AUC:** 0.831  

**Confusion Matrix (test set):**  
[[66 33]]  # TN=66, FP=33
[[ 9 46]] # FN=9, TP=46


### Key Insights

- Random Forest with adjusted threshold balances high recall (0.836) with good F1-Score (0.687), ideal for medical diagnosis.
- Glucose and Glucose_Insulin_Ratio are top predictors, aligning with biological expectations.
- Logistic Regression offers high recall (0.764) for a simpler model, while XGBoost shows potential for further tuning.

## Future Improvements

- Implement cross-validation for robust model evaluation.
- Tune XGBoost hyperparameters (e.g., `learning_rate`, `max_depth`) to improve performance.
- Experiment with additional models (e.g., SVM, neural networks).
- Explore KNN imputation for zeros instead of median.
- Incorporate lifestyle factors if new data is available.
