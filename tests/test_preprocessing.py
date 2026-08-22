import pandas as pd
import pytest

from src.preprocessing import build_preprocessor, split_features_target


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "flow_id": ["FLOW-000001", "FLOW-000002"],
        "flow_duration_s": [4.0, 0.1],
        "total_fwd_packets": [25, 400],
        "total_bwd_packets": [22, 1],
        "total_fwd_bytes": [8000.0, 19000.0],
        "total_bwd_bytes": [9000.0, 60.0],
        "packet_rate": [12.0, 3200.0],
        "byte_rate": [4000.0, 170000.0],
        "avg_packet_size": [400.0, 50.0],
        "fwd_bwd_ratio": [1.1, 180.0],
        "syn_flag_ratio": [0.03, 0.95],
        "ack_flag_ratio": [0.45, 0.01],
        "unique_src_ips": [2, 400],
        "protocol": ["TCP", "TCP"],
        "label": ["BENIGN", "DDOS"],
    })


def test_split_features_target_shapes(sample_df):
    X, y = split_features_target(sample_df)
    assert "label" not in X.columns
    assert "flow_id" not in X.columns
    assert len(y) == len(sample_df)


def test_split_features_target_encodes_binary(sample_df):
    _, y = split_features_target(sample_df)
    assert set(y.unique()).issubset({0, 1})
    assert y.iloc[0] == 0  # BENIGN
    assert y.iloc[1] == 1  # DDOS


def test_preprocessor_fits_and_transforms(sample_df):
    X, _ = split_features_target(sample_df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(sample_df)
    assert transformed.shape[1] > 0


def test_preprocessor_handles_missing_values(sample_df):
    X, _ = split_features_target(sample_df)
    X.loc[0, "avg_packet_size"] = None
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(sample_df)
