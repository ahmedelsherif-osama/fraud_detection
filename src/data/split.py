import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
from configs.config import RAW_DATA_DIR, SPLIT_DATA_DIR
from tools.logger import get_logger

logger = get_logger(__name__)

INGESTED_DATASET_FILENAME = "creditcard_raw_*.csv"


def split_dataset():

    os.makedirs(SPLIT_DATA_DIR, exist_ok=True)

    files = glob.glob(os.path.join(RAW_DATA_DIR, INGESTED_DATASET_FILENAME))

    if not files:
        raise FileNotFoundError("No versioned raw dataset found.")

    raw_dir = Path(RAW_DATA_DIR)
    latest_file = max(raw_dir.iterdir(), key=os.path.getctime)
    logger.info(f"Using dataset: {latest_file}")

    df = pd.read_csv(latest_file)

    logger.info(f"Loaded dataset shape: {df.shape}")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["Class"], random_state=42
    )

    test_df, val_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["Class"], random_state=42
    )

    train_path = os.path.join(SPLIT_DATA_DIR, "train.csv")
    val_path = os.path.join(SPLIT_DATA_DIR, "val.csv")
    test_path = os.path.join(SPLIT_DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print_stats("Train", train_df)
    print_stats("Validation", val_df)
    print_stats("Test", test_df)
    logger.info("Dataset splitting completed.")


def print_stats(name, data):

    logger.info(f"{name} set:")
    logger.info(f"Rows: {len(data)}")

    distribution = data["Class"].value_counts(normalize=True)

    logger.info("Class ratio:")
    logger.info(distribution)


def main():
    split_dataset()


if __name__ == "__main__":
    main()
