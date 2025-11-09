# Weather Prediction using Decision Tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample weather dataset
data = {
    "Temperature": [30, 25, 27, 20, 18, 15, 35, 33, 22, 19],
    "Humidity": [70, 80, 65, 90, 95, 85, 60, 55, 88, 92],
    "Windy": [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    "Play": ["Yes", "No", "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "No"]
}

df = pd.DataFrame(data)
print(df.head())

# Encode categorical column
df["Play"] = df["Play"].map({"Yes": 1, "No": 0})

# Features and target
X = df[["Temperature", "Humidity", "Windy"]]
y = df["Play"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example prediction
new_data = pd.DataFrame({"Temperature": [28], "Humidity": [75], "Windy": [0]})
prediction = model.predict(new_data)
print("Prediction (1=Play, 0=No Play):", prediction[0])
