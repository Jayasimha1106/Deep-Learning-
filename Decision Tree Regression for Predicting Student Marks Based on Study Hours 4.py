import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create dataset
data = {
    "StudyHours": [2, 3, 4, 5, 6, 7],
    "Marks": [35, 45, 55, 65, 75, 85]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and Output
X = df[["StudyHours"]]
y = df["Marks"]

# Create and train Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Predict on training data
pred = model.predict(X)

# Predict marks for 8 study hours
hours = pd.DataFrame({"StudyHours": [8]})
predicted_marks = model.predict(hours)

print("Predicted Marks for 8 hours:", predicted_marks[0])

# Plot Actual Data
plt.scatter(X["StudyHours"], y, color="blue", label="Actual Data")

# Plot Decision Tree Prediction
plt.plot(X["StudyHours"], pred, color="red", marker="o", label="Decision Tree Prediction")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Decision Tree Regression")
plt.legend()
plt.grid(True)

plt.show()