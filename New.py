import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
rainfall = np.random.uniform(0, 200, n_samples)  # Rainfall in mm
soil_moisture = np.random.uniform(0, 1, n_samples)  # Soil moisture (0 to 1)
slope_angle = np.random.uniform(0, 45, n_samples)  # Slope angle in degrees
vibration = np.random.uniform(0, 0.5, n_samples)  # Vibration intensity

# Define landslide occurrence based on thresholds
landslide_occurred = (
    (rainfall > 100) & (soil_moisture > 0.7) & (slope_angle > 30) & (vibration > 0.2)
).astype(int)

# Create DataFrame
data = pd.DataFrame({
    "rainfall": rainfall,
    "soil_moisture": soil_moisture,
    "slope_angle": slope_angle,
    "vibration": vibration,
    "landslide_occurred": landslide_occurred
})

# Save to CSV
data.to_csv("synthetic_landslide_data.csv", index=False)
print("Synthetic dataset saved as 'synthetic_landslide_data.csv'")
