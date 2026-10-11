import csv

from scripts.benchmark import measure
from supportsense.classifier import KeywordClassifier
from supportsense.synthetic import generate_tickets
from training.generate_synthetic_data import main as gen_main, write_csv


async def test_benchmark_measure_reports_sane_numbers():
    texts = [t["text"] for t in generate_tickets(64, seed=2)]
    r = await measure(KeywordClassifier(), texts, batch_size=8, max_wait_ms=5, concurrency=16)
    assert r["tickets_per_sec"] > 0
    assert 0 < r["p50_ms"] <= r["p95_ms"]
    assert 1 <= r["avg_batch"] <= 8


async def test_benchmark_batch_size_one_never_batches():
    texts = [t["text"] for t in generate_tickets(32, seed=2)]
    r = await measure(KeywordClassifier(), texts, batch_size=1, max_wait_ms=0, concurrency=8)
    assert r["avg_batch"] == 1


def test_generate_synthetic_csv(tmp_path):
    out = tmp_path / "d" / "t.csv"
    gen_main(["--n", "30", "--seed", "5", "--out", str(out)])
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 30 and set(rows[0]) == {"text", "label", "urgency"}
    write_csv([{"text": "a", "label": "billing", "urgency": 0.1}], tmp_path / "x.csv")
    assert (tmp_path / "x.csv").exists()
