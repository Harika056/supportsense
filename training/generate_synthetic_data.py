"""Generate SYNTHETIC (template-based) support tickets as CSV: text,label,urgency.

    python -m training.generate_synthetic_data --n 4000 --out data/synthetic_tickets.csv

Synthetic data only exists so the pipeline runs without external datasets. Metrics
measured on it do not say anything about real-world performance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from supportsense.synthetic import generate_tickets


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "urgency"])
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_tickets.csv")
    a = ap.parse_args(argv)
    write_csv(generate_tickets(a.n, a.seed), a.out)
    print(f"Wrote {a.n} SYNTHETIC tickets to {a.out}")


if __name__ == "__main__":
    main()
