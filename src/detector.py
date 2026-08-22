"""
Simulates a live traffic feed and scores each flow through the trained
model in real time, printing alerts when DDoS traffic is detected.

This stands in for a real deployment where flow features would be computed
from live packet capture (e.g. via CICFlowMeter or a custom NetFlow/pcap
exporter) and pushed into this same `score_flow` function.

Run: python -m src.detector
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import ALL_FEATURES
from src.train import load_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
ALERT_THRESHOLD = 0.7  # probability above which a flow is flagged


class DDoSDetector:
    """Wraps a trained pipeline to score individual flows or a stream."""

    def __init__(self, model_path: str | Path = MODEL_PATH, threshold: float = ALERT_THRESHOLD):
        self.pipeline = load_model(model_path)
        self.threshold = threshold

    def score_flow(self, flow: dict) -> dict:
        """Score a single flow (dict of feature_name -> value)."""
        X = pd.DataFrame([flow])[ALL_FEATURES]
        proba = self.pipeline.predict_proba(X)[0, 1]
        return {
            "is_ddos": bool(proba >= self.threshold),
            "confidence": round(float(proba), 4),
        }

    def score_batch(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of flows at once (used by the API and reporting)."""
        X = flows_df[ALL_FEATURES]
        proba = self.pipeline.predict_proba(X)[:, 1]
        result = flows_df.copy()
        result["ddos_confidence"] = np.round(proba, 4)
        result["is_ddos"] = proba >= self.threshold
        return result


def simulate_stream(data_path: str | Path, n_flows: int = 30, delay: float = 0.15):
    """Replay a sample of flows from the dataset as if arriving live."""
    detector = DDoSDetector()
    df = pd.read_csv(data_path).sample(n=n_flows, random_state=None).reset_index(drop=True)

    print(f"Streaming {n_flows} flows through {MODEL_PATH.name} "
          f"(alert threshold = {detector.threshold})\n")

    alerts = 0
    for _, row in df.iterrows():
        flow = row[ALL_FEATURES].to_dict()
        result = detector.score_flow(flow)
        status = "ALERT: DDoS" if result["is_ddos"] else "ok"
        marker = "🚨" if result["is_ddos"] else "  "
        print(
            f"{marker} {row['flow_id']}  proto={flow['protocol']:<4} "
            f"pkt_rate={flow['packet_rate']:>9.1f}/s  "
            f"src_ips={int(flow['unique_src_ips']):>4}  "
            f"confidence={result['confidence']:.2f}  -> {status}"
        )
        if result["is_ddos"]:
            alerts += 1
        time.sleep(delay)

    print(f"\nStream complete: {alerts}/{n_flows} flows flagged as DDoS.")


if __name__ == "__main__":
    simulate_stream(ROOT / "data" / "network_flows.csv")
