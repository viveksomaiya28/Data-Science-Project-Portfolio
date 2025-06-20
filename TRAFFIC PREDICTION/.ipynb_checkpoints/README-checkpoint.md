# Vehicle Traffic Prediction using LSTM

## Project Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict vehicle counts based on historical traffic data. Using lag features of vehicle counts from previous time steps, the model learns temporal patterns to forecast future traffic volumes.

## Dataset
The dataset contains timestamped vehicle counts, with features engineered as lag variables representing vehicle counts at previous 1 to 6 time steps. Missing values were dropped to maintain data integrity.

## Features
- `Vehicles_lag_1` to `Vehicles_lag_6`: Vehicle counts lagged by 1 to 6 time steps.
- `Vehicles`: Actual vehicle count at the current time step (target variable).

## Methodology
1. **Data Preprocessing:**
   - Dropped rows with missing values in selected features and target.
   - Reshaped input features into 3D shape (samples, timesteps, features) required for LSTM.
   - Applied Min-Max scaling separately to features and target.

2. **Model Architecture:**
   - Single LSTM layer with 50 units and ReLU activation.
   - Dense output layer with one neuron for regression output.

3. **Training:**
   - Used Mean Squared Error (MSE) loss function and Adam optimizer.
   - Trained for 20 epochs with 10% validation split.
   - Data split into training and testing sets without shuffling to preserve temporal order.

4. **Evaluation Metrics:**
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

## Results
- Test Loss (MSE): ~0.0013
- MAE: ~3.58 vehicles
- RMSE: ~5.66 vehicles

These results indicate the model captures traffic trends reasonably well, though there is room for improvement with further hyperparameter tuning and feature engineering.

## How to Run
1. Clone the repository.
2. Install dependencies:
3. Prepare your dataset in CSV format as described.
4. Run the notebook or Python script to train and evaluate the model.

## Future Work
- Experiment with deeper LSTM architectures or additional layers.
- Incorporate external factors like weather, holidays, or events.
- Implement cross-validation to better assess generalization.
- Add visualizations comparing actual vs predicted values.

The model developed in this project can accurately forecast vehicle traffic based on historical patterns. On average, the predictions are within 3 to 4 vehicles of the actual counts, which is a strong result for time-series forecasting of this nature.
This means the system can reliably support data-driven traffic planning, such as anticipating peak hours, reducing congestion, and improving resource allocation. While it’s not yet perfect, the model provides a solid starting point that could be expanded by including other factors like weather conditions or public holidays.

*This project demonstrates practical application of deep learning for time series forecasting and can be extended for other traffic-related prediction tasks.*
