import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Create dataset
data = {
    "StudyHours": [2, 3, 4, 5, 6, 7],
    "Result": [0, 0, 1, 1, 1, 1]   # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Input and Output
X = df[["StudyHours"]]
y = df["Result"]

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, y)

# Predict for a student studying 4.5 hours
test_data = pd.DataFrame({"StudyHours": [4.5]})
prediction = model.predict(test_data)

if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")