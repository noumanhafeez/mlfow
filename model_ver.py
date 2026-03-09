import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
data = pd.read_csv("mushrooms.csv")
TARGET_COLUMN = "class"

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression hyperparameter tuning
for max_iter in [50, 100, 150, 200]:
    with mlflow.start_run(run_name=f"LogisticRegression_maxiter_{max_iter}"):
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"LogisticRegression max_iter={max_iter} → Accuracy={acc}")

# Random Forest hyperparameter tuning
from itertools import product
for n_estimators, max_depth in product([50, 100, 150], [5, 10, 15]):
    with mlflow.start_run(run_name=f"RF_{n_estimators}_{max_depth}"):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model", registered_model_name="MushroomClassifier")
        print(f"RF n_estimators={n_estimators} max_depth={max_depth} → Accuracy={acc}")