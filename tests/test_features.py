import pytest
import pandas as pd
from src.features.build_features import split_features_target, scale_features

# Dummy dataset
df = pd.DataFrame(
    {"V1": [0.1, 0.2], "V2": [0.2, 0.3], "Amount": [100, 200], "Class": [0, 1]}
)


def test_split_feature_targets():
    X, y = split_features_target(df)
    assert "Class" not in X.columns
    assert y.name == "Class"
    assert X.shape[0] == y.shape[0]


def test_scale_features_shapes():
    X_train = df.drop(columns=["Class"])
    X_val = X_train.copy()
    X_test = X_train.copy()

    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    # shapes should be preserved
    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape == X_val.shape
    assert X_test_scaled.shape == X_test.shape

    # Amount should be numeric
    assert pd.api.types.is_numeric_dtype(X_train_scaled["Amount"])
