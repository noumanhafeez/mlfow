import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from itertools import product

# Load dataset
data = pd.read_csv("mushrooms.csv")
TARGET_COLUMN = "class"

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment name (all runs will be under this experiment)
mlflow.set_experiment("RandomForest_Experiment")

# Hyperparameter grid
n_estimators_range = [100, 150]
max_depth_range = [5, 10]

for n_estimators, max_depth in product(n_estimators_range, max_depth_range):
    run_name = f"RF_{n_estimators}_{max_depth}"
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log parameters and metric
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # Log model under the same registered model name
        mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForest")

        print(f"Run {run_name} → Accuracy={acc}")

