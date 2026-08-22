from pathlib import Path

import pytest

from src.evaluate import evaluate_all, evaluate_model
from src.train import CANDIDATE_MODELS, make_pipeline, train_test_data

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "network_flows.csv"


@pytest.fixture(scope="module")
def split_data():
    return train_test_data(DATA_PATH)


def test_train_test_data_split_sizes(split_data):
    X_train, X_test, y_train, y_test = split_data
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.02


def test_single_model_trains_and_predicts(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipeline = make_pipeline(CANDIDATE_MODELS["logistic_regression"])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_evaluate_model_returns_expected_keys(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipeline = make_pipeline(CANDIDATE_MODELS["logistic_regression"])
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"):
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_all_ranks_models(split_data):
    X_train, X_test, y_train, y_test = split_data
    fitted = {}
    for name in ("logistic_regression", "random_forest"):
        pipeline = make_pipeline(CANDIDATE_MODELS[name])
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

    results_df = evaluate_all(fitted, X_test, y_test)
    assert len(results_df) == 2
    assert results_df["roc_auc"].is_monotonic_decreasing


def test_model_detects_high_confidence_syn_flood(split_data):
    """Sanity check: an obvious SYN-flood-shaped flow should score high."""
    X_train, X_test, y_train, y_test = split_data
    pipeline = make_pipeline(CANDIDATE_MODELS["random_forest"])
    pipeline.fit(X_train, y_train)

    import pandas as pd
    flow = pd.DataFrame([{
        "flow_duration_s": 0.05, "total_fwd_packets": 500, "total_bwd_packets": 0,
        "total_fwd_bytes": 30000.0, "total_bwd_bytes": 0.0, "packet_rate": 9000.0,
        "byte_rate": 550000.0, "avg_packet_size": 60.0, "fwd_bwd_ratio": 500.0,
        "syn_flag_ratio": 0.99, "ack_flag_ratio": 0.0, "unique_src_ips": 450,
        "protocol": "TCP",
    }])
    proba = pipeline.predict_proba(flow)[0, 1]
    assert proba > 0.8
