import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("Loading Iris dataset...")
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/results.json").write_text(
    json.dumps(
        {"accuracy": accuracy, "test_samples": len(y_test)},
        indent=2
    )
)

print(f"Accuracy: {accuracy:.4f}")
print("Saved artifacts/results.json")
