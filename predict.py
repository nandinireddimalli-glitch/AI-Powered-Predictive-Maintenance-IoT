from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import clean_new_readings

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "predictive_maintenance.joblib"

def predict_readings(readings, threshold=0.50):
    """Predict failure risk for incoming sensor records."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run python main.py first.")
    model = joblib.load(MODEL_PATH)
    features = clean_new_readings(readings)
    probabilities = model.predict_proba(features)[:, 1]
    results = readings.copy().reset_index(drop=True)
    results["failure_probability"] = probabilities
    results["predicted_failure"] = (probabilities >= threshold).astype(int)
    results["alert"] = results["failure_probability"].apply(
        lambda p: "INSPECT" if p >= threshold else "NORMAL"
    )
    return results

def create_virtual_readings(df, count=20, seed=42):
    """Sample historical sensor measurements as a virtual inference feed."""
    if count < 1:
        raise ValueError("count must be at least 1.")
    cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    rng = np.random.default_rng(seed)
    sample = df[cols].sample(
        n=count, replace=count > len(df), random_state=seed
    ).reset_index(drop=True)
    sample.insert(0, "device_id", [f"IOT-MACHINE-{i:03d}" for i in range(1, count + 1)])
    sample.insert(1, "reading_id", [f"READING-{i:04d}" for i in range(1, count + 1)])
    sample["Air temperature [K]"] += rng.normal(0, 0.15, count)
    sample["Process temperature [K]"] += rng.normal(0, 0.15, count)
    sample["Torque [Nm]"] += rng.normal(0, 0.30, count)
    return sample
