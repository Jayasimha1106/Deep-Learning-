import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create dataset
data = {
    "StudyHours": [2, 3, 4, 5, 6, 7],
    "Marks": [35, 45, 55, 65, 75, 85]
}

df = pd.DataFrame(data)

# Input and Output
X = df[["StudyHours"]]
y = df["Marks"]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict(X)

# Equation
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict marks for 8 study hours
hours = [[8]]
print("Predicted Marks for 8 hours:", model.predict(hours)[0])

# Plot
plt.scatter(X, y, color="blue")
plt.plot(X, pred, color="red")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.show()