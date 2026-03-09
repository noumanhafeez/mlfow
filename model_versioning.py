import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def tracking_pipeline():

    # =============================
    # 1️⃣ Load Dataset
    # =============================
    data = pd.read_csv("mushrooms.csv")
    TARGET_COLUMN = "class"

    # =============================
    # 2️⃣ Encode Data
    # =============================

    encoders = {}

    for column in data.columns:
        if data[column].dtype == "object":
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            encoders[column] = le

    # =============================
    # 3️⃣ Train Test Split
    # =============================

    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =============================
    # 4️⃣ Define Multiple Models
    # =============================

    models = {
        "GaussianNB": GaussianNB(),

        "LogisticRegression": LogisticRegression(
            max_iter=200
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10
        )
    }

    # =============================
    # 5️⃣ Train Each Model
    # =============================

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            # Log params
            mlflow.log_param("model_type", model_name)

            # Log metric
            mlflow.log_metric("accuracy", accuracy)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="MushroomClassifier"
            )

            print(f"{model_name} accuracy:", accuracy)


tracking_pipeline()