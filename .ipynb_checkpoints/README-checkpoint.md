# Data-science-project-portfolio

A collection of real-world machine learning projects covering predictive maintenance (with a dashboard), customer churn prediction, diabetes risk prediction, and time series forecasting — with both traditional ML and deep learning models, plus web app deployment.

---

## 1. Turbofan Jet Engine RUL Prediction (NASA C-MAPSS)

**Goal**: Predict Remaining Useful Life (RUL) of jet engines using sensor data.

**Dataset**: NASA C-MAPSS FD001

**Techniques**:
- Random Forest Regressor
- Rolling window features (`mean`, `std`, window=5)
- MinMaxScaler normalization
- Strong EDA (sensor degradation trends, correlations)

**Key Metrics**:
| Dataset      | RMSE (cycles) | R² Score |
|--------------|---------------|----------|
| Validation   | 34.24         | 0.74     |
| Test         | 34.24         | 0.32     |

**Highlights**:
- Top feature: `sensor4_roll_mean` (Importance: 46.5%)
- Feature importance visualized
- Output files: `train_final.csv`, `test_final.csv`

[Google Drive – Dataset & Files](https://drive.google.com/drive/folders/15auZkVkV6vAInt_z9dIgOI-IFea-Xtn3?usp=drive_link)
> Large files (model, scaler, etc.) are externally hosted due to GitHub size limits.
---

## 2. Diabetes Risk Prediction using Machine Learning

**Goal**: Predict diabetes risk using clinical and demographic data.

**Dataset**: Diabetes dataset (e.g., Pima Indians or similar)

**Techniques**:
- Exploratory Data Analysis (EDA) covering distribution and correlation of features
- Feature engineering and handling missing values
- Trained models: Logistic Regression, Random Forest, XGBoost
- Model evaluation via precision, recall, F1-score, ROC AUC

**Performance**:
- ROC AUC scores ranging ~0.80–0.85 depending on model
- Feature importance insights to identify key risk factors

**Highlights**:
- Detailed step-by-step preprocessing pipeline
- Balanced class weights or sampling techniques to handle imbalance
- Clear code explanations for each step to aid learning

---

## 3. Telco Customer Churn Prediction

**Goal**: Predict which telecom customers are likely to churn.

**Dataset**: Telco Customer Churn Dataset

**Models**:
- Random Forest (ROC AUC: 0.84, Recall: 77% for churn)
- Logistic Regression (ROC AUC: 0.839)

**Preprocessing**:
- 20 selected features (raw + engineered)
- `StandardScaler` for numerical, `OneHotEncoder` for categorical
- Balanced class weights, stratified train-test split

**Evaluation**:
- Confusion matrix, classification report, ROC AUC
- Random Forest captured nonlinear interactions better

**Next Steps**:
- SHAP integration for explainability
- Gradient boosting or neural network exploration

---

## 4. Vehicle Traffic Prediction using LSTM

**Goal**: Predict vehicle count using time series lag features.

**Model**: LSTM Neural Network  
**Input Features**: `Vehicles_lag_1` to `Vehicles_lag_6`

**Preprocessing**:
- Scaled with `MinMaxScaler`
- Reshaped into LSTM-compatible 3D array

**Architecture**:
- LSTM (50 units) → Dense (1)
- Trained with Adam optimizer and MSE loss

**Metrics**:
- RMSE: 5.66  
- MAE: 3.58

**Use Case**: Urban traffic forecasting and infrastructure planning

---

## Tools & Libraries Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**, **TensorFlow/Keras**, **Matplotlib**, **Seaborn**
- **Streamlit** for deployment
- **RandomizedSearchCV** for tuning


