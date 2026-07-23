import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Dataset
data = {
    "Study_Hours": [2, 3, 5, 6, 8],
    "Attendance": [60, 65, 80, 85, 95],
    "Assignment": [50, 55, 70, 80, 90],
    "Result": [0, 0, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Inputs
X = df[["Study_Hours", "Attendance", "Assignment"]]

# Output
y = df["Result"]

# Create MLP Classifier
model = MLPClassifier(
    hidden_layer_sizes=(4,),
    max_iter=2000,
    random_state=42
)

# Train the model
model.fit(X_scaled, y)

# New student data
new_student = pd.DataFrame({
    "Study_Hours": [4],
    "Attendance": [75],
    "Assignment": [65]
})

# Scale the new data
new_student_scaled = scaler.transform(new_student)

# Predict
prediction = model.predict(new_student_scaled)

# Display result
if prediction[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")