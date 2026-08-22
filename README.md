# DDoS Detector

An ML-based network intrusion detection system that classifies network
traffic flows as **benign** or **DDoS**, built with **pandas** and
**scikit-learn**, served through a **FastAPI** inference API with a live
**React** monitoring dashboard, and fully **Dockerized**.

> **Note on data:** this project ships with a synthetic-but-realistic flow
> dataset (`data/generate_traffic_data.py`) so it runs fully offline. It
> models three real DDoS patterns — SYN flood, UDP flood, and HTTP flood —
> plus benign traffic *including a "flash crowd" case deliberately built to
> overlap with HTTP flood*, so the classifier faces a genuinely hard
> boundary rather than trivially separable classes. Point `main.py` at a
> real flow export (e.g. from CICFlowMeter or a NetFlow/IPFIX collector)
> with the same column names and the rest of the pipeline is unchanged.

## Problem

DDoS attacks flood a target with traffic to exhaust its resources. This
project frames detection as **binary classification on network flow
features** (packet rate, byte rate, SYN flag ratio, source IP diversity,
etc.) — the same flow-level approach used by real-world tools like
CICFlowMeter-based IDS systems.

## Project Structure

```
ddos-detector/
├── data/
│   ├── generate_traffic_data.py   # synthetic flow dataset generator + live-flow simulator
│   └── network_flows.csv          # generated dataset (6,000 flows)
├── src/
│   ├── preprocessing.py            # feature schema + sklearn preprocessing pipeline
│   ├── train.py                    # candidate models + training
│   ├── evaluate.py                 # metrics computation
│   ├── visualize.py                # EDA and results plots
│   └── detector.py                 # DDoSDetector class + live stream simulator
├── api/
│   └── app.py                      # FastAPI service (/predict, /predict/batch, /metrics, /stream/next, /simulate/{category}, /health)
├── frontend/
│   ├── src/                        # React (Vite) live monitoring dashboard
│   ├── Dockerfile                  # multi-stage build -> nginx
│   └── nginx.conf                  # serves the SPA, proxies /api/* to the backend
├── tests/
│   ├── test_preprocessing.py
│   ├── test_train_evaluate.py
│   └── test_api.py
├── results/
│   ├── metrics.json
│   └── figures/
├── models/                         # saved trained models (.joblib, gitignored)
├── main.py                         # end-to-end training pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Approach

1. **Data generation** — 6,000 synthetic flows: benign, flash-crowd
   (hard negative), SYN flood, UDP flood, HTTP flood — with injected
   measurement noise so classes overlap realistically.
2. **EDA** — label distribution, packet rate by traffic type, source IP
   diversity by traffic type (see `results/figures/`).
3. **Preprocessing** — a `ColumnTransformer`: median-impute + scale numeric
   flow features, most-frequent-impute + one-hot-encode protocol.
4. **Modeling** — three candidates compared: Logistic Regression, Random
   Forest, Gradient Boosting.
5. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC on a
   stratified 20% held-out test set.
6. **Serving** — the best model is wrapped in a `DDoSDetector` class used
   by both a live-stream simulator and a FastAPI inference endpoint.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting** | 0.997 | 0.993 | 0.998 | 0.995 | **1.000** |
| Random Forest | 0.997 | 0.993 | 0.998 | 0.995 | 0.9999 |
| Logistic Regression | 0.996 | 1.000 | 0.988 | 0.994 | 0.9999 |

![Model Comparison](results/figures/model_comparison.png)
![ROC Curve](results/figures/roc_curve.png)
![Confusion Matrix](results/figures/confusion_matrix.png)

### Key EDA findings

- DDoS flows show packet rates an order of magnitude above benign traffic,
  even accounting for legitimate flash-crowd spikes.
- Attack flows draw from far more unique source IPs (spoofed/botnet), a
  strong discriminating signal on its own.

![Packet Rate by Label](results/figures/packet_rate_by_label.png)
![Unique Source IPs](results/figures/unique_src_ips.png)

## Setup

```bash
git clone <this-repo-url>
cd ddos-detector
pip install -r requirements.txt
```

## Usage

**Train the models and generate results:**

```bash
python data/generate_traffic_data.py   # regenerate the dataset (optional, already included)
python main.py                          # trains all models, saves metrics + figures
```

**Run the live detector simulation** (streams sample flows through the
trained model and prints real-time alerts):

```bash
python -m src.detector
```

**Run the API locally:**

```bash
uvicorn api.app:app --reload --port 8000
```

Then try it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "flow_duration_s": 0.12, "total_fwd_packets": 350, "total_bwd_packets": 1,
    "total_fwd_bytes": 19500.0, "total_bwd_bytes": 60.0, "packet_rate": 2900.0,
    "byte_rate": 160000.0, "avg_packet_size": 55.0, "fwd_bwd_ratio": 175.0,
    "syn_flag_ratio": 0.95, "ack_flag_ratio": 0.02, "unique_src_ips": 340,
    "protocol": "TCP"
  }'
```

Interactive API docs: `http://localhost:8000/docs`

**Run tests:**

```bash
pytest tests/ -v
```

## Docker

The image trains the model at build time (fixed random seed → reproducible
build), so the API is ready to serve as soon as the container starts.

```bash
docker compose up --build
```

This starts two services:

- **API** — `http://localhost:8000` (docs at `/docs`)
- **Dashboard** — `http://localhost:3000` — live monitoring UI: a rolling
  traffic feed, packet-rate chart, model comparison chart, and simulation
  controls, all pulling from the API's `/stream/next`, `/simulate/{category}`,
  and `/metrics` endpoints (nginx proxies `/api/*` from the dashboard
  container to the backend container, so no CORS or extra config needed).

By default the dashboard auto-generates random traffic (weighted like the
training data) every ~1.2s. Use **Pause auto traffic** to stop that, then
use the **Inject: ...** buttons to manually trigger a specific flow type
(benign, flash crowd, SYN flood, UDP flood, HTTP flood) on demand — useful
for demoing the detector without relying on random chance.

To also run the live-alert stream simulator (console output) against the
same trained model:

```bash
docker compose --profile detector up --build
```

Or run the API alone with plain Docker:

```bash
docker build -t ddos-detector .
docker run -p 8000:8000 ddos-detector
```

## Dashboard (local dev, without Docker)

```bash
cd frontend
npm install
npm run dev       # http://localhost:3000, talks directly to the API on :8000
```

Make sure the API is running first (`uvicorn api.app:app --port 8000`).

## Tech Stack

- **pandas** / **numpy** — data manipulation and synthetic flow generation
- **scikit-learn** — preprocessing pipelines, models, metrics
- **matplotlib** / **seaborn** — visualization
- **FastAPI** / **uvicorn** — inference API
- **React** / **Vite** / **recharts** — live monitoring dashboard
- **pytest** / **httpx** — unit and API testing
- **Docker** / **docker-compose** / **nginx** — containerized deployment

## Possible Extensions

- Replace synthetic data with a real captured/exported flow dataset
- Add an unsupervised anomaly detector (Isolation Forest) for zero-day attack patterns
- Stream real packet captures through `DDoSDetector.score_flow` via a NetFlow/pcap exporter
- Add Prometheus metrics + Grafana dashboard for alert volume over time
- Auto-mitigation hook (e.g. trigger a firewall rule) on sustained high-confidence alerts

## License

MIT — see [LICENSE](LICENSE).
