"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Model file not found. Please place the best churn model at {MODEL_PATH}"
    ) from exc


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of feature values and returns a churn prediction (0 or 1).
    """
    features_array = np.asarray(features, dtype=float).reshape(1, -1)
    prediction = model.predict(features_array)
    return int(prediction[0])


if __name__ == "__main__":
    sample = [0.0] * model.n_features_in_
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
