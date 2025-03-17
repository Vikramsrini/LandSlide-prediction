# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the dataset
# Replace 'landslide_data.csv' with your dataset file
# Dataset should have columns: rainfall, soil_moisture, slope_angle, vibration, landslide_occurred
data = pd.read_csv("synthetic_landslide_data.csv")

# Step 2: Data Preprocessing
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Fill missing values (if any)
data.fillna(data.mean(), inplace=True)

# Features (X) and Target (y)
X = data[["rainfall", "soil_moisture", "slope_angle", "vibration"]]
y = data["landslide_occurred"]

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: Save the trained model for deployment
joblib.dump(model, "landslide_prediction_model.pkl")
print("Model saved as 'landslide_prediction_model.pkl'")

# Step 7: Example of making a prediction with new data
# Replace these values with real-time sensor data
new_data = {
    "rainfall": [50.0],  # Rainfall in mm
    "soil_moisture": [0.6],  # Soil moisture (0 to 1)
    "slope_angle": [30.0],  # Slope angle in degrees
    "vibration": [0.1],  # Vibration intensity
}

# Convert to DataFrame
new_df = pd.DataFrame(new_data)

# Predict landslide probability
prediction = model.predict(new_df)
print("Prediction (0 = No Landslide, 1 = Landslide):", prediction)

# Predict landslide probability (probability output)
probability = model.predict_proba(new_df)
print("Landslide Probability:", probability[0][1])
