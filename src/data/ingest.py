import os
import hashlib
from datetime import datetime
import pandas as pd
from tools.logger import get_logger
from configs.config import RAW_DATA_DIR
import json


logger = get_logger(__name__)

INPUT_FILENAME = "creditcard.csv"


REQUIRED_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Class",
]

def compute_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def validate_file(path: str):
    """Check if file exists"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    logger.info(f"Dataset found at {path}")


def load_data(path: str):
    """Load CSV into pandas dataframe."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df


def validate_columns(df: pd.DataFrame):
    """Check that all required columns exist."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.isnull().sum().sum() > 0:
        logger.warning("Missing values detected")
    logger.info("All required columns present")


def save_versioned_copy_with_metadata(df: pd.DataFrame, input_path: str):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_file = os.path.join(RAW_DATA_DIR, f"creditcard_raw_{timestamp}.csv")
    metadata_file = os.path.join(RAW_DATA_DIR, f"creditcard_raw_{timestamp}.json")

    df.to_csv(data_file, index=False)

    file_hash = compute_hash(input_path)

    metadata = {
        "source": os.path.basename(input_path),
        "rows": df.shape[0],
        "columns_count": df.shape[1],
        "columns": df.columns.tolist(),
        "timestamp": timestamp,
        "file_hash": file_hash,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved dataset to {data_file}")
    logger.info(f"Saved metadata to {metadata_file}")


def print_dataset_stats(df: pd.DataFrame):
    """Print basic statistics."""
    logger.info("Dataset Statistics:")
    logger.info(f"Number of rows: {df.shape[0]}")
    logger.info(f"Number of columns: {df.shape[1]}")
    logger.info("Class distribution:")
    logger.info(df["Class"].value_counts())
    fraud_ratio = df["Class"].mean()
    if fraud_ratio < 0.01:
        logger.warning("Severe class imbalance detected")
    logger.info("Class ratio (fraud / non-fraud):")
    logger.info(df["Class"].value_counts(normalize=True))


def ingest_data():
    input_path = os.path.join(RAW_DATA_DIR, INPUT_FILENAME)
    validate_file(input_path)
    compute_hash(input_path)
    df = load_data(input_path)
    validate_columns(df)
    save_versioned_copy_with_metadata(df, input_path)
    print_dataset_stats(df)


def main():
    ingest_data()


if __name__ == "__main__":
    main()
