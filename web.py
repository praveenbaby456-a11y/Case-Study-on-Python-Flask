from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

REQUIRED_FIELDS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

NUMERIC_FIELDS = {
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences"
}


def _load_model():
    if not MODEL_PATH.exists():
        from model import train_and_save_model
        train_and_save_model(MODEL_PATH)

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    if not hasattr(model, "named_steps"):
        from model import train_and_save_model
        model = train_and_save_model(MODEL_PATH)

    return model


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    try:
        if not form_data:
            raise ValueError("No form data was submitted.")

        payload = {}
        for field in REQUIRED_FIELDS:
            raw_value = form_data.get(field)
            if raw_value is None or raw_value == "":
                raise ValueError(f"Please enter a value for {field}.")
            payload[field] = int(raw_value) if field in NUMERIC_FIELDS else raw_value

        model = _load_model()
        prediction = model.predict(pd.DataFrame([payload], columns=REQUIRED_FIELDS))[0]

        return render_template("index.html", form_data=form_data, prediction=prediction)
    except Exception as exc:
        return render_template("index.html", form_data=form_data, error=f"Unable to predict: {exc}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
