from pathlib import Path
from src.data_loader import load_dataset
from src.model import train_models
from src.predict import create_virtual_readings, predict_readings
from src.visualize import create_visualizations

ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 65)
    print("AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR IoT DEVICES")
    print("=" * 65)
    df = load_dataset()
    print("\\nDataset preview:")
    print(df.head())
    print("\\nTarget distribution:")
    print(df["Machine failure"].value_counts(dropna=False))

    model, X_test, y_test, metrics, best_name = train_models(df)
    create_visualizations(metrics, model, X_test, y_test, best_name)

    virtual_readings = create_virtual_readings(df, count=20)
    predictions = predict_readings(virtual_readings, threshold=0.50)
    out_dir = ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "virtual_sensor_predictions.csv"
    predictions.to_csv(output_path, index=False)

    display_columns = [
        "device_id", "reading_id", "Air temperature [K]",
        "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "failure_probability", "predicted_failure", "alert",
    ]
    print("\\nVirtual sensor predictions:")
    print(predictions[display_columns].to_string(index=False))
    print("\\nAlert summary:")
    print(predictions["alert"].value_counts())
    print(f"\\nSaved predictions: {output_path}")
    print("Project completed successfully.")

if __name__ == "__main__":
    main()
