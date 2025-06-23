# Predictive Maintenance with NASA Turbofan Jet Engine Dataset
## File Links
https://drive.google.com/drive/folders/15auZkVkV6vAInt_z9dIgOI-IFea-Xtn3?usp=drive_link
- The datasets are in the 'data' folder, once you click on the link.

## Overview

This project predicts the **Remaining Useful Life (RUL)** of turbofan jet engines using the **NASA C-MAPSS FD001** dataset. A **Random Forest** regression model estimates how many operational cycles remain before an engine fails, based on sensor data. The model is deployed as a **Streamlit web app**, allowing users to:

- Input **normalized sensor values (0–1)** manually  
- Load **sample test data**  
- Predict **RUL instantly**  
- View a **feature importance plot**

> Note: This is a working prototype. Test set performance indicates room for improvement.

---

## Key Metrics

| Dataset          | RMSE (cycles) | R² Score |
|------------------|---------------|----------|
| Validation Set   | 34.24         | 0.74     |
| Test Set (Proto) | 34.24         | 0.32     |

- **Top Feature**: `sensor4_roll_mean` (Importance: 46.5%)

## Features

- Enter **44 normalized sensor values (0–1)**
- Load **sample data** from `test_final.csv`
- Display **predicted RUL**
- Visualize **feature importance**

---

## Data Preprocessing

**Dataset Used**: NASA Turbofan Engine Degradation Simulation (FD001)

### Files:
- `train_FD001.txt`: Training dataset  
- `test_FD001.txt`: Test dataset  
- `RUL_FD001.txt`: True RUL values for test engines

### Processing Steps:
1. Removed low-variance sensors (e.g., `sensor1`, `sensor5`)
2. Added RUL values to training data
3. Created rolling features (mean and std, window = 5 cycles) for 14 sensors
4. Normalized features using `MinMaxScaler` (range 0–1)

### Output Files:
- `train_final.csv`: Processed training data  
- `test_final.csv`: Processed test data  

---

## Exploratory Data Analysis (EDA)

### Sensor Trends
- `sensor3` and `sensor4` showed zig-zag degradation patterns

### Correlation with RUL:

| Sensor   | Correlation |
|----------|-------------|
| sensor4  | -0.679      |
| sensor11 | -0.696      |
| sensor12 |  0.672      |

### RUL Distribution
- Highly skewed  
- Mean ≈ 108 cycles  
- Max = 361 cycles  

---

## Modeling

- **Algorithm**: Random Forest Regressor  
- **Features Used**: 44 (sensor values, settings, rolling stats)  
- **Training Data**: `train_final.csv`

### Performance:

| Dataset    | RMSE   | R²   |
|------------|--------|------|
| Validation | 34.24  | 0.74 |
| Test       | 34.24  | 0.32 |

### Feature Importance:

| Feature              | Importance |
|----------------------|------------|
| sensor4_roll_mean    | 46.5%      |
| sensor9_roll_mean    | 8.2%       |
| sensor21_roll_mean   | 7.34%      |


## Deployment

- **Platform**: Streamlit Cloud  
- **App File**: `app.py`

### App Features:
- Input sensor data manually or load sample data
- View predicted RUL
- Display feature importance plot

> Note: Large files (model, scaler, etc.) are hosted externally due to GitHub file size limits. See below for download links.

---

## File Links
https://drive.google.com/drive/folders/15auZkVkV6vAInt_z9dIgOI-IFea-Xtn3?usp=drive_link
- The datasets are in the 'data' folder, once you click on the link.
