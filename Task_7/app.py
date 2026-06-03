from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "theft_rate_model.pkl"

app = Flask(__name__)


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


artifact = None
load_error = None
try:
    artifact = load_artifact()
except Exception as exc:
    load_error = str(exc)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None
    entered_values = None
    error_message = load_error

    if request.method == "POST":
        if artifact is None:
            error_message = load_error or "Model is not available."
        else:
            try:
                feature_columns = artifact["feature_columns"]
                input_data = {
                    "Foot_Traffic": float(request.form["foot_traffic"]),
                    "Staff_Count": float(request.form["staff_count"]),
                    "Hour": float(request.form["hour"]),
                    "High_Value_Items": float(request.form["high_value_items"]),
                    "Shrinkage_INR": float(request.form["shrinkage_inr"]),
                }
                entered_values = input_data
                input_frame = pd.DataFrame([input_data], columns=feature_columns)

                model = artifact["model"]
                label_encoder = artifact["label_encoder"]

                predicted_class = model.predict(input_frame)[0]
                prediction = label_encoder.inverse_transform([predicted_class])[0]

                probability_values = model.predict_proba(input_frame)[0]
                probabilities = [
                    {"label": class_name, "value": probability_values[index]}
                    for index, class_name in enumerate(label_encoder.classes_)
                ]
            except Exception as exc:
                error_message = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probabilities,
        entered_values=entered_values,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
