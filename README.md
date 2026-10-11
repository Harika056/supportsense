# SupportSense

Real-time support-ticket triage: a DistilBERT multi-task classifier (8 categories + urgency score) served by a FastAPI service with asyncio micro-batching, a Kafka-based streaming pipeline, and a live WebSocket dashboard.

## What is verified and what is not
- Unit/integration tests run in CI (Python 3.11). Tests needing torch are skipped when it is not installed.
- The **keyword** backend is for tests and demos only. It is not a trained model.
- The bundled data generator is **synthetic**. Metrics from training on it (even very high accuracy) are not real-world performance. Use a real labelled ticket dataset (see below) before quoting any accuracy.
- Kafka consumer/producer and `docker-compose.yml` were not run by the author. Treat them as untested until you run them.

## Quick start (no Docker, no Kafka)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```
Terminal 1 (dashboard):
```
uvicorn dashboard.app:create_app --factory --port 8001
```
Open http://localhost:8001. Terminal 2 (feeds fake tickets through the keyword classifier):
```
$env:MODEL_BACKEND="keyword"
python -m scripts.run_local_pipeline --rate 5 --count 200
```

## Train the real model
```
pip install -r requirements-train.txt
python -m training.train --num-synthetic 2000 --epochs 2
```
Writes `artifacts/model` and `artifacts/metrics.json` (which records whether the data was synthetic).

### Using a real dataset
```
python -m scripts.prepare_csv --input my.csv --text-col text --label-col category --list-labels
python -m scripts.prepare_csv --input my.csv --text-col text --label-col category --map "Billing issue=billing" --out data/tickets.csv
python -m training.train --data data/tickets.csv --epochs 2
```

## Serve and benchmark
```
$env:MODEL_BACKEND="transformers"; $env:MODEL_DIR="artifacts\model"
uvicorn service.app:create_app --factory --port 8000
python -m scripts.benchmark --tickets 512 --concurrency 64 --batch-size 16
```
Endpoints: `POST /classify`, `POST /classify/batch`, `GET /health`, `GET /metrics`.

## Results (fill only from your own runs)
| Metric | Value | Notes |
|---|---|---|
| Tests passing | | `56 passed, 1 skipped` |
| Accuracy / macro-F1 (real data) | | `artifacts/metrics.json`, real dataset only |
| Throughput, no batching | | `scripts.benchmark` |
| Throughput, micro-batching | | `scripts.benchmark` |
| p95 latency | | `scripts.benchmark` |

## Kafka (untested)
`docker compose up --build` starts Kafka (KRaft), service, dashboard, consumer and producer. Requires a trained `artifacts/model`.
