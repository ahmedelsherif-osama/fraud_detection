# imports
import os
from xgboost import XGBClassifier
import pandas as pd
import joblib

# config / paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_processed.csv")
VAL_FILE = os.path.join(PROCESSED_DIR, "val_processed.csv")


# functions
def load_train_val():
    # read csv fo rboth train and val
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)

    # combin train and val
    combined = pd.concat([train, val], axis=0).reset_index(drop=True)

    # define x and y from combined dataset
    X = combined.drop(columns=["Class"])
    y = combined["Class"]

    # print confirmation
    print(f"[INFO] Combined train+val shape: {X.shape}")
    # return x y
    return X, y


def train_final_model(X, y, best_params=None):
    if best_params is None:
        best_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

    model = XGBClassifier(**best_params)

    print("[INFO] Training final model on train+val...")
    model.fit(X, y, verbose=50)
    print("[INFO] Final model training complete.")
    return model


def save_model(model):
    # timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # new model path
    model_path = os.path.join(ARTIFACTS_DIR, f"xgb_final_model_{timestamp}.joblib")

    # save model/dump
    joblib.dump(model, model_path)

    # print confirmation
    print(f"[INFO] Final model saved to: {model_path}")

    # return model path
    return model_path


# main
def train_final():
    # load data to x and y
    X, y = load_train_val()
    # train final model
    model = train_final_model(X, y)
    # save model
    save_model(model)


def main():
    train_final()


if __name__ == "__main__":
    main()
