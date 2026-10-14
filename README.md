# AI-Powered Predictive Maintenance System for IoT Devices

A student-friendly, industry-inspired predictive maintenance prototype using the UCI AI4I 2020 dataset, Python, and scikit-learn. It trains classifiers, evaluates failure detection, simulates incoming sensor readings, assigns illustrative inspection alerts, and saves charts and CSV reports.

> **Scope and limitations:** AI4I 2020 is a synthetic dataset. This repository demonstrates an end-to-end workflow; it is not connected to physical IoT hardware and is not validated for safety-critical maintenance decisions. Alerts are model outputs for demonstration only.

## Problem statement
Unexpected equipment failures can cause downtime, maintenance expense, and production disruption. Predictive maintenance uses machine condition data to estimate failure risk so teams can investigate potential issues earlier.

## Features
- Automatic dataset download from the official UCI repository
- Data validation and temperature-difference feature engineering
- Logistic Regression and Random Forest comparison
- Class-aware metrics: precision, recall, F1 and ROC-AUC
- Held-out test set, with validation-based model selection
- Saved model pipeline and prediction CSVs
- Confusion matrix, metric comparison, feature importance, and risk chart
- Virtual sensor feed with generated device/reading IDs and slight measurement noise

## Technology stack
Python 3.11, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib.

## Dataset
UCI AI4I 2020 Predictive Maintenance Dataset:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset has 10,000 synthetic records. Predictors include product type, air/process temperature, rotational speed, torque, and tool wear. The target is `Machine failure`.

Failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers are excluded from the predictors to avoid leakage.

## Architecture
```text
AI4I CSV / virtual readings
        |
        v
Validation + feature engineering
        |
        v
Preprocessing pipeline + classifier
        |
        +--> evaluation metrics and charts
        |
        v
Failure probability + illustrative alert
        |
        v
CSV predictions / saved model
```

## Setup

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked for the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
From the project root:
```bash
python main.py
```

The first run needs internet access to download the dataset. If automatic download fails, download the archive from the official UCI link above and put its CSV in `data/raw/`, then run again.

## Outputs
- `models/predictive_maintenance.joblib` — fitted preprocessing + classifier pipeline
- `outputs/metrics/model_comparison.csv` — held-out test metrics
- `outputs/predictions/test_predictions.csv` — test predictions
- `outputs/predictions/evaluation_predictions.csv` — evaluation probabilities
- `outputs/predictions/virtual_sensor_predictions.csv` — simulated incoming readings and alerts
- `images/model_comparison.png`
- `images/confusion_matrix.png`
- `images/feature_importance.png` — generated when the selected model supports feature importance
- `images/failure_risk.png`

Generated data, models and outputs are ignored by Git by default. Choose a few useful, non-sensitive charts to publish intentionally.

## Evaluation
Accuracy alone is insufficient for rare failures. Review precision, recall, F1-score, ROC-AUC and the confusion matrix. False negatives represent missed failures; false positives represent unnecessary inspections.

Model selection uses a validation split within the training data. The held-out test set is reserved for final comparison.

## Screenshots
After running the project, add selected images to this README, for example:
```markdown
![Model comparison](images/model_comparison.png)
![Confusion matrix](images/confusion_matrix.png)
![Failure risk](images/failure_risk.png)
```

## Industry relevance
This prototype illustrates the data-to-alert workflow used in condition monitoring. A production deployment would additionally require validated real-machine data, time-aware testing, sensor-quality checks, monitoring for data drift, security, maintenance workflows, and engineering approval of alert thresholds.

## Learning outcomes
- Data loading, validation, and feature engineering
- Supervised classification on imbalanced labels
- Leakage-aware preprocessing pipelines
- Model evaluation and interpretation
- Virtual sensor inference and alert generation
- Reproducible project structure and GitHub documentation

## Future improvements
- Add time-series / run-to-failure datasets
- Add a Streamlit dashboard and simulated MQTT stream
- Tune alert thresholds using validation costs
- Add tests, CI, drift monitoring, and model versioning
- Evaluate on real, authorized equipment data before operational use

## License
Check the UCI dataset's terms before redistributing dataset files. Project code may be published under a license of your choice.
