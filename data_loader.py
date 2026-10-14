from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

def load_dataset():
    """Download, extract, and load the official AI4I CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        zip_path = RAW_DIR / "ai4i_dataset.zip"
        print("Downloading the UCI AI4I dataset...")
        try:
            urlretrieve(DATA_URL, zip_path)
        except Exception as exc:
            raise RuntimeError(
                "Dataset download failed. Check your internet connection, "
                "or download the archive from https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset "
                "and place its CSV in data/raw/."
            ) from exc
        with ZipFile(zip_path, "r") as archive:
            members = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
                and not name.startswith("__MACOSX/")
            ]
            if not members:
                raise FileNotFoundError("No CSV file found in the downloaded archive.")
            for member in members:
                target = RAW_DIR / Path(member).name
                with archive.open(member) as source:
                    target.write_bytes(source.read())
        csv_files = list(RAW_DIR.glob("*.csv"))
    preferred = [p for p in csv_files if "ai4i" in p.name.lower()]
    csv_path = preferred[0] if preferred else csv_files[0]
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"Dataset loaded: {csv_path.name} | rows={len(df):,} | columns={len(df.columns)}")
    return df

if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
    print("\\nColumns:", data.columns.tolist())
