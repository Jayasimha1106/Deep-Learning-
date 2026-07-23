import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Dataset
data = {
    "StudyHours": [2, 3, 4, 5, 6, 7],
    "Marks": [35, 45, 55, 65, 75, 85]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and Output
X = df[["StudyHours"]]
y = df["Marks"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Error Analysis
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("Actual Marks      :", list(y))
print("Predicted Marks   :", y_pred)

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)