import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/dataset/landmarks_dataset.csv", header=None)

# First column = label
y = data.iloc[:, 0]

# Remaining 63 columns = features
X = data.iloc[:, 1:]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training RandomForest model...")

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/landmark_model.pkl")
print("Landmark model saved!")
