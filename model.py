import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "student-data.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

CATEGORICAL_COLUMNS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]
NUMERIC_COLUMNS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences"
]


def _build_preprocessor():
    try:
        categorical_encoder = OneHotEncoder(drop="first", sparse_output=False)
    except TypeError:
        categorical_encoder = OneHotEncoder(drop="first", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_encoder, CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def train_and_save_model(model_path: str | Path = MODEL_PATH):
    student_data = pd.read_csv(DATA_PATH)
    x = student_data.drop(columns=["passed"])
    y = student_data["passed"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    preprocessor = _build_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(x_train, y_train)
    accuracy = pipeline.score(x_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")

    with Path(model_path).open("wb") as file:
        pickle.dump(pipeline, file)

    print(f"Model training completed and saved as {model_path}")
    return pipeline


if __name__ == "__main__":
    train_and_save_model(MODEL_PATH)