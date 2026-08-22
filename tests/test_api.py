import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


@pytest.fixture
def syn_flood_flow():
    return {
        "flow_duration_s": 0.12, "total_fwd_packets": 350, "total_bwd_packets": 1,
        "total_fwd_bytes": 19500.0, "total_bwd_bytes": 60.0, "packet_rate": 2900.0,
        "byte_rate": 160000.0, "avg_packet_size": 55.0, "fwd_bwd_ratio": 175.0,
        "syn_flag_ratio": 0.95, "ack_flag_ratio": 0.02, "unique_src_ips": 340,
        "protocol": "TCP",
    }


@pytest.fixture
def benign_flow():
    return {
        "flow_duration_s": 4.0, "total_fwd_packets": 25, "total_bwd_packets": 22,
        "total_fwd_bytes": 8000.0, "total_bwd_bytes": 9000.0, "packet_rate": 12.0,
        "byte_rate": 4000.0, "avg_packet_size": 400.0, "fwd_bwd_ratio": 1.1,
        "syn_flag_ratio": 0.03, "ack_flag_ratio": 0.45, "unique_src_ips": 2,
        "protocol": "TCP",
    }


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_flags_syn_flood(syn_flood_flow):
    r = client.post("/predict", json=syn_flood_flow)
    assert r.status_code == 200
    body = r.json()
    assert body["is_ddos"] is True
    assert body["confidence"] > 0.7


def test_predict_passes_benign(benign_flow):
    r = client.post("/predict", json=benign_flow)
    assert r.status_code == 200
    body = r.json()
    assert body["is_ddos"] is False


def test_predict_rejects_invalid_protocol(benign_flow):
    bad_flow = {**benign_flow, "protocol": "NOT_A_PROTOCOL"}
    r = client.post("/predict", json=bad_flow)
    assert r.status_code == 422


def test_predict_batch(syn_flood_flow, benign_flow):
    r = client.post("/predict/batch", json={"flows": [syn_flood_flow, benign_flow]})
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 2
    assert results[0]["is_ddos"] is True
    assert results[1]["is_ddos"] is False


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "best_model" in body


def test_stream_next_returns_scored_flow():
    r = client.get("/stream/next")
    assert r.status_code == 200
    body = r.json()
    for key in ("timestamp", "protocol", "packet_rate", "unique_src_ips", "is_ddos", "confidence", "threshold"):
        assert key in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_simulate_each_category():
    for category in ("benign", "flash_crowd", "syn_flood", "udp_flood", "http_flood"):
        r = client.get(f"/simulate/{category}")
        assert r.status_code == 200
        assert r.json()["category"] == category


def test_simulate_attack_categories_flagged_as_ddos():
    for category in ("syn_flood", "udp_flood", "http_flood"):
        r = client.get(f"/simulate/{category}")
        assert r.json()["is_ddos"] is True


def test_simulate_invalid_category_rejected():
    r = client.get("/simulate/not_a_real_category")
    assert r.status_code == 422
