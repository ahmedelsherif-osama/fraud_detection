import pandas as pd
from src.data.ingest import ingest_data
from src.inference.predict import predict
from src.features.build_features import split_features_target, scale_features


def test_csv_reading():
    df = pd.read_csv("data/raw/creditcard.csv")
    assert not df.empty


def test_feature_split():
    df = pd.read_csv("data/processed/train_processed.csv")
    X, y = split_features_target(df)
    assert "Class" not in X.columns
    assert y.name == "Class"


def test_scaling():
    df = pd.read_csv("data/processed/train_processed.csv")
    X, y = split_features_target(df)
    X_train, X_val, X_test = scale_features(X, X, X)
    assert "Amount" in X_train.columns


def test_prediction():
    sample = pd.DataFrame(
        [
            {
                "Time": 10000,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536346,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62,
            }
        ]
    )
    result = predict(sample)
    assert isinstance(result, list)
    assert "fraud_probability" in result[0]
