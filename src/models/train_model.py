import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


# 1. determine paths: base, processed, artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# 2. make dir of artifacts if it doesnt exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_processed_data():
    """Load processed feature datasets: train, val"""
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val_processed.csv"))

    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]

    X_val = val.drop(columns=["Class"])
    y_val = val["Class"]

    return X_train, y_train, X_val, y_val


def train_xgb_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with early stopping"""
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print("\n[INFO] Validation Classification Report:")
    print(classification_report(y_val, y_pred))
    print(f"[INFO] Validation ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")


def save_model(model):
    """Save model to artifacts folder with timestamp"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(ARTIFACTS_DIR, f"xgb_model_{timestamp}.joblib")
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to: {model_path}")
    return model_path


def train_model():
    # 3. load processed data to x, y train and val only
    X_train, y_train, X_val, y_val = load_processed_data()

    # 4. define the model using train_xgb
    model = train_xgb_model(X_train, y_train, X_val, y_val)

    # 5. evaluate the model using model and val datasets
    evaluate_model(model, X_val, y_val)

    # 6. save model
    save_model(model)


def main():
    train_model()


if __name__ == "__main__":
    main()
