import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 6, 7],
    'Attendance': [45, 50, 60, 65, 70, 75, 80, 90, 55, 72, 78, 85],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['StudyHours', 'Attendance']]
y = df['Pass']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Neural Network Model
model = MLPClassifier(
    hidden_layer_sizes=(4,),   # One hidden layer with 4 neurons
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print Results
print("Predicted Values :", y_pred)
print("Actual Values    :", y_test.values)
print("Accuracy         :", accuracy_score(y_test, y_pred))

# Predict for a new student
new_student = [[5, 75]]   # StudyHours=5, Attendance=75
new_student = scaler.transform(new_student)

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("New Student Prediction: PASS")
else:
    print("New Student Prediction: FAIL")