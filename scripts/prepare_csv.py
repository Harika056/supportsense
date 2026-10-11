"""Turn any ticket CSV into the format training expects (text,label[,urgency]).

Public support-ticket datasets use their own category names. Map them onto the 8 labels
this project uses, drop what does not fit, and write a clean CSV.

    python -m scripts.prepare_csv --input raw.csv --text-col "Ticket Description" ^
        --label-col "Ticket Type" --map "Billing inquiry=billing" --map "Refund request=refund_return" ^
        --out data/tickets.csv

Rows whose label is not in --map are dropped (the count is printed). Run with --list-labels first
to see the labels your file contains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from supportsense.labels import CATEGORIES


def prepare(df: pd.DataFrame, text_col: str, label_col: str, mapping: dict[str, str],
            urgency_col: str | None = None) -> pd.DataFrame:
    for col in [text_col, label_col] + ([urgency_col] if urgency_col else []):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not found; columns are {list(df.columns)}")
    bad = sorted(set(mapping.values()) - set(CATEGORIES))
    if bad:
        raise ValueError(f"unknown target labels {bad}; choose from {CATEGORIES}")
    out = pd.DataFrame({"text": df[text_col].astype(str).str.strip(), "label": df[label_col].map(mapping)})
    if urgency_col:
        out["urgency"] = pd.to_numeric(df[urgency_col], errors="coerce").clip(0, 1)
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out


def parse_map(items: list[str]) -> dict[str, str]:
    mapping = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--map expects 'source=target', got {item!r}")
        src, dst = item.rsplit("=", 1)
        mapping[src.strip()] = dst.strip()
    return mapping


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--urgency-col", default=None)
    ap.add_argument("--map", action="append", default=[], help="source_label=target_label (repeatable)")
    ap.add_argument("--out", default="data/tickets.csv")
    ap.add_argument("--list-labels", action="store_true", help="print the labels in --label-col and exit")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.input)
    if a.list_labels:
        if a.label_col not in df.columns:
            sys.exit(f"column {a.label_col!r} not found; columns are {list(df.columns)}")
        print(df[a.label_col].value_counts().to_string())
        return
    try:
        out = prepare(df, a.text_col, a.label_col, parse_map(a.map), a.urgency_col)
    except ValueError as exc:
        sys.exit(str(exc))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"read {len(df)} rows, wrote {len(out)} rows to {a.out}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
