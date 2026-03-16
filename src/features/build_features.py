import os
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from tools.logger import get_logger

logger = get_logger(__name__)


# CONFIG

## Define needed paths/directories

BASE_DIR = os.path.join(os.path.dirname(__file__), "../../")

PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
SPLIT_DIR = os.path.join(BASE_DIR, "data/split")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

## create these directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


# FUNCTIONS
def load_datasets():
    # read from csv all 3 split datasets: train, val, test
    train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    # print info loaded confirmation & shapes for all 3
    print("[INFO] Loaded datasets:")
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    # return all 3 df's
    return train, val, test


def split_features_target(df: pd.DataFrame):
    # define x and y from dataframe
    ## for x just drop class column
    X = df.drop(columns=["Class"])
    ## for y take only class column
    y = df["Class"]

    # return x and y
    return X, y


def scale_features(X_train, X_val, X_test):
    # define/initialize scaler
    scaler = RobustScaler()

    # ensure Amount is numeric
    X_train["Amount"] = pd.to_numeric(X_train["Amount"], errors="coerce")
    X_val["Amount"] = pd.to_numeric(X_val["Amount"], errors="coerce")
    X_test["Amount"] = pd.to_numeric(X_test["Amount"], errors="coerce")

    # fit/transform train, val, test on specific needed columns

    scaler = RobustScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_val["Amount"] = scaler.transform(X_val[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    # save scaler for later inference
    ## define path for saving
    scaler_path = os.path.join(ARTIFACT_DIR, "amount_scaler.joblib")
    ## save file/scaler for later re-use
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved successfully")

    # print info confirmation scaler saved to path
    print(f"[INFO] Scaler saved to: {scaler_path}")

    # return new scaled X datasets train, test, val
    return X_train, X_val, X_test


def save_processed(X, y, name):
    df = X.copy()
    df["Class"] = y

    output_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df.to_csv(output_path, index=False)

    print(f"[INFO] Saved processed dataset: {output_path}")


# MAIN FEATURE PIPELINE
def build_features():

    # load datasets - train, val, test
    train, val, test = load_datasets()

    # split features target - (X, y) for: train, val, test
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(val)
    X_test, y_test = split_features_target(test)

    # scale features - all inputs X: train, val, test
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    # save processed sets(Scaled for now): train, val, test
    save_processed(X_train, y_train, "train_processed")
    save_processed(X_val, y_val, "val_processed")
    save_processed(X_test, y_test, "test_processed")

    # print/log confirmation message (Feature eng compl)
    print("\n[INFO] Feature engineering completed.")


# ENTRY POINT


def main():
    build_features()


if __name__ == "__main__":
    main()
