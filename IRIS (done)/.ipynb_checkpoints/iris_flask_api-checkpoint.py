# iris_flask_api.py

# ---------------------------
# Step 1: Import libraries
# ---------------------------
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# ---------------------------
# Step 2: Load saved artifacts
# ---------------------------
# Load trained Logistic Regression model
lr_model = joblib.load("final_lr_model.pkl")

# Load scaler used during training
scaler = joblib.load("scaler.pkl")

# Load LabelEncoder to decode numeric predictions to species names
label_encoder = joblib.load("label_encoder.pkl")

# ---------------------------
# Step 3: Initialize Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Step 4: Home route (optional)
# ---------------------------
@app.route('/')
def home():
    return "Iris Logistic Regression API is running!"

# ---------------------------
# Step 5: Prediction route
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1️⃣ Receive input JSON
        data = request.get_json()  # expects a list of dicts

        # 2️⃣ Convert JSON to DataFrame
        df = pd.DataFrame(data)

        # 3️⃣ Scale features using the saved scaler
        df_scaled = scaler.transform(df)

        # 4️⃣ Predict numeric class labels
        preds_numeric = lr_model.predict(df_scaled)

        # 5️⃣ Decode numeric labels to species names
        preds_labels = label_encoder.inverse_transform(preds_numeric)

        # 6️⃣ Return predictions as JSON
        return jsonify(preds_labels.tolist())

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------------
# Step 6: Run the app
# ---------------------------
if __name__ == "__main__":
    # use_reloader=False avoids SystemExit in Jupyter environments
    app.run(debug=True, use_reloader=False)
