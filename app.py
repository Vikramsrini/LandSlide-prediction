from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This allows requests from any origin. For production, restrict it to specific domains.

# Load the trained model
model = joblib.load("landslide_prediction_model.pkl")

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get data from request
    data = request.json

    # Extract features
    rainfall = data.get("rainfall")
    soil_moisture = data.get("soil_moisture")
    slope_angle = data.get("slope_angle")
    vibration = data.get("vibration")

    # Validate input data
    if None in (rainfall, soil_moisture, slope_angle, vibration):
        return jsonify({"error": "Missing required fields"}), 400

    # Preprocess data
    input_data = np.array([[rainfall, soil_moisture, slope_angle, vibration]])

    # Make prediction
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return result
    return jsonify({
        "prediction": int(prediction[0]),
        "probability": float(probability)
    })

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
