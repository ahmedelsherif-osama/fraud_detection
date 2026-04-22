import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "split")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

for path in [DATA_DIR, ARTIFACTS_DIR, RAW_DATA_DIR, SPLIT_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(path, exist_ok=True)
