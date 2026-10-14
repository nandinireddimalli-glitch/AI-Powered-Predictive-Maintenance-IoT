from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
sns.set_theme(style="whitegrid")

def create_visualizations(metrics, best_model, X_test, y_test, best_name):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).T.plot(kind="bar", figsize=(10, 6))
    plt.title("Predictive Maintenance Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison.png", dpi=150)
    plt.close()

    predictions = best_model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No failure", "Failure"],
                yticklabels=["No failure", "Failure"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    classifier = best_model.named_steps["classifier"]
    preprocessor = best_model.named_steps["preprocessor"]
    if hasattr(classifier, "feature_importances_"):
        names = preprocessor.get_feature_names_out()
        importance = pd.Series(classifier.feature_importances_, index=names).sort_values(ascending=False).head(10)
        plt.figure(figsize=(9, 6))
        importance.sort_values().plot(kind="barh")
        plt.title("Top 10 Random Forest Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=150)
        plt.close()

    probabilities = best_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({
        "actual_failure": y_test.to_numpy(),
        "predicted_failure": predictions,
        "failure_probability": probabilities,
    }).to_csv(PREDICTIONS_DIR / "evaluation_predictions.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(probabilities)), probabilities, marker=".", linestyle="none",
             alpha=0.7, label="Predicted failure probability")
    plt.axhline(0.50, linestyle="--", label="Illustrative 0.50 threshold")
    plt.xlabel("Held-out test record index")
    plt.ylabel("Predicted failure probability")
    plt.title("Failure Risk Across Test Records")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "failure_risk.png", dpi=150)
    plt.close()
    print("Charts saved to images/.")
