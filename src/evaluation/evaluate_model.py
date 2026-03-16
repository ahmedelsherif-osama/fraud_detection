import os
import glob
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# path config

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# functions


def load_latest_model():
    """Load most recent trained model from artifacts"""
    model_files = glob.glob(os.path.join(ARTIFACTS_DIR, "xgb_model_*.joblib"))

    if not model_files:
        raise FileNotFoundError("No trained model found in artifacts")

    latest_model = max(model_files, key=os.path.getctime)
    print(f"[INFO] Loading model: {latest_model}")

    model = joblib.load(latest_model)
    return model


def load_test_data():
    """Load processed test dataset."""
    test_path = os.path.join(PROCESSED_DIR, "test_processed.csv")
    df = pd.read_csv(test_path)
    X_test = df.drop(columns=["Class"])
    y_test = df["Class"]

    print("[INFO] Loaded test dataset:", df.shape)

    return X_test, y_test


def evaluate_model():
    model = load_latest_model()

    X_test, y_test = load_test_data()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n[INFO] Test Classification Report:")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_proba)
    print(f"[INFO] Test ROC-AUC: {roc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\n[INFO] Confusion Matrix:")
    print(cm)

    print()


def main():
    evaluate_model()


if __name__ == "__main__":
    main()
