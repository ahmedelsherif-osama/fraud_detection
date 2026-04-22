import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "split")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

print(DATA_DIR)
