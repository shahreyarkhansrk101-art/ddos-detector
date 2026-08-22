"""Data loading and preprocessing for the DDoS flow classifier."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "label"
ID_COLUMN = "flow_id"

NUMERIC_FEATURES = [
    "flow_duration_s",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "packet_rate",
    "byte_rate",
    "avg_packet_size",
    "fwd_bwd_ratio",
    "syn_flag_ratio",
    "ack_flag_ratio",
    "unique_src_ips",
]
CATEGORICAL_FEATURES = ["protocol"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from the binary label (1 = DDOS)."""
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = (df[TARGET_COLUMN] == "DDOS").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Median-impute + scale numeric flow features; most-frequent-impute +
    one-hot-encode the protocol field.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
