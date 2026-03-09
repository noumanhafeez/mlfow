import mlflow
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def tracking_pipeline():

    # =============================
    # 1️⃣ Load Dataset
    # =============================
    data = pd.read_csv("mushrooms.csv")

    # 🔹 Change this if your target column has different name
    TARGET_COLUMN = "class"

    # =============================
    # 2️⃣ Encode Categorical Columns
    # =============================

    encoders = {}  # store encoders for each column

    for column in data.columns:
        if data[column].dtype == "object":
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            encoders[column] = le

    # =============================
    # 3️⃣ Split Features & Target
    # =============================

    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="experiment_tracking"):
        model = GaussianNB()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_type", "GaussianNB")
        mlflow.log_param("test_size", "0.2")
        mlflow.log_param("encoding", "LabelEncoder")
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        print("Model Trained Successfully")
        print("Accuracy:", accuracy)
    return accuracy


experiment = tracking_pipeline()
