from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, prepare_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"

def build_preprocessor(scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

def train_models(df):
    X, y, _ = prepare_data(df)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Both target classes need at least two examples.")
    # Hold out test data once. Select using a validation split within training data.
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.20,
        random_state=43, stratify=y_train_all
    )
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
    }
    fitted, validation_f1 = {}, {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        validation_f1[name] = f1_score(y_val, val_pred, zero_division=0)
        fitted[name] = model
        print(f"Validation F1: {validation_f1[name]:.4f}")

    best_name = max(validation_f1, key=validation_f1.get)
    best_model = fitted[best_name]
    # Refit selected pipeline on all training data, preserving held-out test set.
    best_model.fit(X_train_all, y_train_all)
    metrics = {}
    all_test_predictions = {}
    for name, model in fitted.items():
        if name == best_name:
            model = best_model
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        all_test_predictions[name] = (pred, proba)
        print(f"\\n{name} held-out test metrics:")
        for key, value in metrics[name].items():
            print(f"  {key}: {value:.4f}")
        print(classification_report(y_test, pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, pred, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "predictive_maintenance.joblib")
    pd.DataFrame(metrics).T.to_csv(METRICS_DIR / "model_comparison.csv")
    pred, proba = all_test_predictions[best_name]
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": pred,
        "failure_probability": proba,
    }).to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)
    print(f"\\nSelected model (validation F1): {best_name}")
    print("Saved model to models/predictive_maintenance.joblib")
    return best_model, X_test, y_test, metrics, best_name
