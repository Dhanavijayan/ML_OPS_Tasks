from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "supermarket_sample.csv"
MODEL_DIR = BASE_DIR / "artifacts"
MODEL_PATH = MODEL_DIR / "theft_rate_model.pkl"

FEATURE_COLUMNS = [
    "Foot_Traffic",
    "Staff_Count",
    "Hour",
    "High_Value_Items",
    "Shrinkage_INR",
]
TARGET_COLUMN = "Theft_Level"


def train_and_save_model() -> Path:
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y_encoded)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(artifact, MODEL_PATH)
    return MODEL_PATH


if __name__ == "__main__":
    saved_path = train_and_save_model()
    print(f"Model saved to: {saved_path}")
