import pandas as pd

TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Temperature difference [K]",
]
CATEGORICAL_FEATURES = ["Type"]

def prepare_data(df):
    """Validate records and return feature matrix, target, and cleaned frame."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", TARGET,
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Machine failure must contain only 0 or 1.")
    df["Temperature difference [K]"] = (
        pd.to_numeric(df["Process temperature [K]"], errors="raise")
        - pd.to_numeric(df["Air temperature [K]"], errors="raise")
    )
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y, df

def clean_new_readings(df):
    """Prepare unlabeled sensor readings with the same feature definitions."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Sensor columns missing: {sorted(missing)}")
    df["Temperature difference [K]"] = (
        pd.to_numeric(df["Process temperature [K]"], errors="raise")
        - pd.to_numeric(df["Air temperature [K]"], errors="raise")
    )
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("Sensor readings contain missing numeric values.")
    if not df["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Type must be one of L, M, or H.")
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
