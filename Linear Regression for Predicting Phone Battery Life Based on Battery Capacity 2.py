import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create dataset
data = {
    "BatteryCapacity": [3000, 3500, 4000, 4500, 5000, 5500],
    "BatteryLife": [8, 10, 12, 14, 16, 18]
}

df = pd.DataFrame(data)

# Input (X) and Output (y)
X = df[["BatteryCapacity"]]
y = df["BatteryLife"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict battery life for 6000 mAh
prediction = model.predict([[6000]])

print("Predicted Battery Life:", prediction[0], "hours")

# Plot graph
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Battery Capacity (mAh)")
plt.ylabel("Battery Life (Hours)")
plt.title("Linear Regression - Phone Battery")
plt.show()