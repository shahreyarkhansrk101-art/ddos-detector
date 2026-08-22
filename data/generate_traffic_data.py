"""
Generates a synthetic network-flow dataset with realistic benign traffic
and three common DDoS attack patterns (SYN flood, UDP flood, HTTP flood),
so the ML pipeline runs fully offline with genuine signal to learn from.

Features are flow-level, the same style used by real tools like
CICFlowMeter: duration, packet/byte rates, flag counts, source diversity.

Run: python data/generate_traffic_data.py
Output: data/network_flows.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_FLOWS = 6000


def _benign_flows(n, rng):
    duration = rng.exponential(4.0, n).clip(0.05, 60)
    fwd_packets = rng.poisson(25, n).clip(1)
    bwd_packets = rng.poisson(22, n).clip(0)
    fwd_bytes = fwd_packets * rng.normal(500, 150, n).clip(40, 1500)
    bwd_bytes = bwd_packets * rng.normal(600, 200, n).clip(40, 1500)
    syn_ratio = rng.uniform(0.0, 0.08, n)
    ack_ratio = rng.uniform(0.35, 0.55, n)
    unique_src_ips = rng.integers(1, 4, n)  # normal clients rarely multiplex IPs
    protocol = rng.choice(["TCP", "UDP", "ICMP"], n, p=[0.75, 0.22, 0.03])
    return _assemble(duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
                      syn_ratio, ack_ratio, unique_src_ips, protocol, "BENIGN", rng)


def _flash_crowd(n, rng):
    """Legitimate traffic spike (e.g. a viral link) — deliberately overlaps
    with HTTP flood so the classifier faces a genuinely hard boundary."""
    duration = rng.exponential(1.5, n).clip(0.1, 15)
    fwd_packets = rng.poisson(120, n).clip(5)
    bwd_packets = rng.poisson(115, n).clip(0)
    fwd_bytes = fwd_packets * rng.normal(480, 120, n).clip(100, 1500)
    bwd_bytes = bwd_packets * rng.normal(350, 120, n).clip(50, 1500)
    syn_ratio = rng.uniform(0.05, 0.25, n)
    ack_ratio = rng.uniform(0.35, 0.55, n)
    unique_src_ips = rng.integers(20, 90, n)  # many real users, but not botnet-scale
    protocol = np.full(n, "TCP")
    return _assemble(duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
                      syn_ratio, ack_ratio, unique_src_ips, protocol, "BENIGN", rng)


def _syn_flood(n, rng):
    duration = rng.exponential(0.15, n).clip(0.001, 2)
    fwd_packets = rng.poisson(400, n).clip(10)
    bwd_packets = rng.poisson(1, n)  # server barely responds
    fwd_bytes = fwd_packets * rng.normal(60, 10, n).clip(40, 100)  # tiny SYN packets
    bwd_bytes = bwd_packets * rng.normal(60, 10, n).clip(0, 100)
    syn_ratio = rng.uniform(0.85, 1.0, n)
    ack_ratio = rng.uniform(0.0, 0.05, n)
    unique_src_ips = rng.integers(50, 500, n)  # spoofed / botnet sources
    protocol = np.full(n, "TCP")
    return _assemble(duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
                      syn_ratio, ack_ratio, unique_src_ips, protocol, "DDOS", rng)


def _udp_flood(n, rng):
    duration = rng.exponential(0.3, n).clip(0.001, 3)
    fwd_packets = rng.poisson(800, n).clip(20)
    bwd_packets = rng.poisson(0.5, n)
    fwd_bytes = fwd_packets * rng.normal(1200, 300, n).clip(64, 1500)  # volumetric
    bwd_bytes = bwd_packets * rng.normal(60, 10, n).clip(0, 100)
    syn_ratio = np.zeros(n)  # UDP has no SYN flags
    ack_ratio = np.zeros(n)
    unique_src_ips = rng.integers(30, 300, n)
    protocol = np.full(n, "UDP")
    return _assemble(duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
                      syn_ratio, ack_ratio, unique_src_ips, protocol, "DDOS", rng)


def _http_flood(n, rng):
    duration = rng.exponential(2.0, n).clip(0.1, 20)
    fwd_packets = rng.poisson(150, n).clip(5)
    bwd_packets = rng.poisson(140, n).clip(0)  # server does respond, just overwhelmed
    fwd_bytes = fwd_packets * rng.normal(450, 100, n).clip(100, 1500)
    bwd_bytes = bwd_packets * rng.normal(300, 100, n).clip(50, 1500)
    syn_ratio = rng.uniform(0.05, 0.2, n)
    ack_ratio = rng.uniform(0.4, 0.6, n)
    unique_src_ips = rng.integers(80, 400, n)  # botnet of "real" clients
    protocol = np.full(n, "TCP")
    return _assemble(duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
                      syn_ratio, ack_ratio, unique_src_ips, protocol, "DDOS", rng)


def _assemble(duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes,
              syn_ratio, ack_ratio, unique_src_ips, protocol, label, rng=None):
    total_packets = fwd_packets + bwd_packets
    total_bytes = fwd_bytes + bwd_bytes
    packet_rate = total_packets / duration
    byte_rate = total_bytes / duration
    avg_packet_size = total_bytes / total_packets.clip(min=1)
    fwd_bwd_ratio = (fwd_packets + 1) / (bwd_packets + 1)

    # Measurement noise: real flow exporters and mixed network conditions
    # blur the boundary between attack and legitimate-but-heavy traffic,
    # so classifiers see realistic overlap rather than a clean separation.
    if rng is not None:
        n = len(duration)
        packet_rate = packet_rate * rng.normal(1.0, 0.18, n).clip(0.5, 1.8)
        byte_rate = byte_rate * rng.normal(1.0, 0.18, n).clip(0.5, 1.8)
        syn_ratio = np.clip(syn_ratio + rng.normal(0, 0.08, n), 0, 1)
        ack_ratio = np.clip(ack_ratio + rng.normal(0, 0.08, n), 0, 1)
        unique_src_ips = np.clip(
            unique_src_ips + rng.normal(0, unique_src_ips.mean() * 0.15 + 1, n), 1, None
        ).astype(int)

    return pd.DataFrame({
        "flow_duration_s": duration,
        "total_fwd_packets": fwd_packets,
        "total_bwd_packets": bwd_packets,
        "total_fwd_bytes": fwd_bytes,
        "total_bwd_bytes": bwd_bytes,
        "packet_rate": packet_rate,
        "byte_rate": byte_rate,
        "avg_packet_size": avg_packet_size,
        "fwd_bwd_ratio": fwd_bwd_ratio,
        "syn_flag_ratio": syn_ratio,
        "ack_flag_ratio": ack_ratio,
        "unique_src_ips": unique_src_ips,
        "protocol": protocol,
        "label": label,
    })


def generate_dataset(n=N_FLOWS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_benign = int(n * 0.55)
    n_flash = int(n * 0.10)
    n_attack = n - n_benign - n_flash
    n_syn = n_attack // 3
    n_udp = n_attack // 3
    n_http = n_attack - n_syn - n_udp

    parts = [
        _benign_flows(n_benign, rng),
        _flash_crowd(n_flash, rng),
        _syn_flood(n_syn, rng),
        _udp_flood(n_udp, rng),
        _http_flood(n_http, rng),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "flow_id", [f"FLOW-{i:06d}" for i in range(1, len(df) + 1)])

    # small amount of realistic missingness
    missing_idx = rng.choice(len(df), size=int(len(df) * 0.015), replace=False)
    df.loc[missing_idx, "avg_packet_size"] = np.nan

    return df


CATEGORY_GENERATORS = {
    "benign": _benign_flows,
    "flash_crowd": _flash_crowd,
    "syn_flood": _syn_flood,
    "udp_flood": _udp_flood,
    "http_flood": _http_flood,
}
RANDOM_CATEGORY_WEIGHTS = [0.55, 0.10, 0.117, 0.117, 0.116]


def generate_flow(category: str, rng: np.random.Generator) -> dict:
    """Generate a single flow of a specific category (e.g. for a manual
    'inject this attack type' trigger)."""
    if category not in CATEGORY_GENERATORS:
        raise ValueError(f"Unknown category: {category}. Choose from {list(CATEGORY_GENERATORS)}")
    row = CATEGORY_GENERATORS[category](1, rng).iloc[0]
    return row.drop(labels=[c for c in ("label",) if c in row.index]).to_dict() | {
        "true_label": row["label"],
        "category": category,
    }


def generate_random_flow(rng: np.random.Generator) -> dict:
    """
    Generate a single random flow (any traffic category), weighted the same
    way as the full dataset. Used by the API's /stream/next endpoint to
    simulate a live feed without needing a real packet capture.
    """
    category = rng.choice(list(CATEGORY_GENERATORS), p=RANDOM_CATEGORY_WEIGHTS)
    return generate_flow(category, rng)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).parent / "network_flows.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} flows to {out_path}")
    print(df["label"].value_counts())
