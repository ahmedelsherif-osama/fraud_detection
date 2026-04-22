import os
import glob
import joblib
import pandas as pd
import argparse


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_latest_model():
    """Load the most recent trained model."""
    model_files = glob.glob(os.path.join(ARTIFACTS_DIR, "xgb_final_model_*.joblib"))

    if not model_files:
        raise FileNotFoundError("No trained model found in artifacts directory.")

    latest_model = max(model_files, key=os.path.getctime)

    print(f"[INFO] Loading model: {latest_model}")

    model = joblib.load(latest_model)
    return model


def load_scaler():
    """Load saved feature scaler."""
    scaler_path = os.path.join(ARTIFACTS_DIR, "amount_scaler.joblib")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found in artifacts directory.")

    print(f"[INFO] Loading scaler: {scaler_path}")

    scaler = joblib.load(scaler_path)
    return scaler


def preprocess_input(df: pd.DataFrame):
    """
    Apply the same preprocessing used during training.
    """
    scaler = load_scaler()

    if "Amount" in df.columns:
        df["Amount"] = scaler.transform(df[["Amount"]])

    return df


def predict(transaction_df: pd.DataFrame, threshold: float = 0.5):
    """
    Predict fraud probability and class.

    Args:
        transaction_df: DataFrame containing transaction features
        threshold: classification threshold

    Returns:
        dict containing probability and prediction
    """

    model = load_latest_model()

    transaction_df = preprocess_input(transaction_df)

    fraud_proba = model.predict_proba(transaction_df)[:, 1]
    fraud_pred = (fraud_proba >= threshold).astype(int)

    results = []

    for proba, pred in zip(fraud_proba, fraud_pred):
        results.append(
            {"fraud_probability": float(proba), "fraud_prediction": int(pred)}
        )

    return results


def run_cli():
    parser = argparse.ArgumentParser(description="Fraud detection inference")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file containing transactions",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fraud classification threshold",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print(f"[INFO] Loaded input data: {df.shape}")

    results = predict(df, threshold=args.threshold)

    output_df = pd.DataFrame(results)

    print("\nPredictions:")
    print(output_df.head())

    output_path = args.input.replace(".csv", "_predictions.csv")
    output_df.to_csv(output_path, index=False)

    print(f"\n[INFO] Predictions saved to: {output_path}")


def main():
    run_cli()


if __name__ == "__main__":
    main()
