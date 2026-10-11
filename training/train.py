"""Fine-tune DistilBERT (category + urgency heads) with a plain PyTorch loop.

Examples
    python -m training.train                          # synthetic data, distilbert-base-uncased
    python -m training.train --data my_tickets.csv --epochs 3 --batch-size 16
    python -m training.train --tiny-offline --max-steps 1   # offline plumbing smoke test only

Evaluation: stratified train/val/test split. The validation split is only logged
per epoch; the reported metrics (metrics.json, confusion matrix) are computed ONCE
on the held-out TEST split after training. Nothing is tuned on test.

--data CSV columns: text,label[,urgency]. label must be one of supportsense.labels.CATEGORIES;
urgency is optional (0-1); rows without it are masked out of the urgency loss.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from supportsense.labels import CATEGORIES, LABEL2ID
from supportsense.synthetic import generate_tickets
from training.model import MultiTaskDistilBert


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="CSV with text,label[,urgency]. Default: in-memory SYNTHETIC data")
    ap.add_argument("--num-synthetic", type=int, default=4000, help="rows of synthetic data when --data is omitted")
    ap.add_argument("--out-dir", default="artifacts/model")
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--urgency-weight", type=float, default=1.0)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--tiny-offline", action="store_true",
                    help="random-init tiny DistilBERT + locally built vocab; no downloads. Plumbing check only.")
    return ap.parse_args(argv)


def load_frame(args) -> tuple[pd.DataFrame, bool]:
    synthetic = args.data is None
    if synthetic:
        print("No --data given: using SYNTHETIC template tickets (pipeline demo only).")
        df = pd.DataFrame(generate_tickets(args.num_synthetic, args.seed))
    else:
        df = pd.read_csv(args.data)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required columns: {sorted(missing)}")
    if "urgency" not in df.columns:
        df["urgency"] = np.nan
    unknown = sorted(set(df["label"]) - set(CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown labels {unknown}; map them to {CATEGORIES} first.")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    return df, synthetic


def split_frame(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_tiny_offline(texts: list[str], out_dir: Path):
    """Tiny random-init DistilBERT and a WordPiece-less vocab built from the data. No network."""
    from transformers import BertTokenizer, DistilBertConfig, DistilBertModel

    words = sorted({w for t in texts for w in t.lower().replace(".", " ").replace(",", " ").split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizer(str(vocab_path), do_lower_case=True)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=2, n_heads=2, max_position_embeddings=512)
    return DistilBertModel(cfg), tok


class Collate:
    def __init__(self, tokenizer, max_length):
        self.tok, self.max_length = tokenizer, max_length

    def __call__(self, rows):
        enc = self.tok([r["text"] for r in rows], padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        enc["labels"] = torch.tensor([LABEL2ID[r["label"]] for r in rows])
        urg = torch.tensor([r["urgency"] if r["urgency"] == r["urgency"] else -1.0 for r in rows], dtype=torch.float)
        enc["urgency"] = urg  # -1 marks "no label"
        return enc


def predict(model, loader, device):
    model.eval()
    preds, gold, up, ug = [], [], [], []
    with torch.no_grad():
        for b in loader:
            logits, urg = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            preds += logits.argmax(-1).cpu().tolist()
            gold += b["labels"].tolist()
            up += urg.cpu().tolist()
            ug += b["urgency"].tolist()
    return np.array(gold), np.array(preds), np.array(ug), np.array(up)


def urgency_mae(ug, up):
    m = ug >= 0
    return float(np.abs(ug[m] - up[m]).mean()) if m.any() else None


def save_confusion_png(y, p, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y, p, labels=list(range(len(CATEGORIES))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45, ha="right")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (held-out TEST split)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, synthetic = load_frame(args)
    train_df, val_df, test_df = split_frame(df, args.val_size, args.test_size, args.seed)
    print(f"split sizes train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}  device={device}")

    if args.tiny_offline:
        encoder, tokenizer = build_tiny_offline(df["text"].tolist(), out_dir)
    else:
        from transformers import AutoTokenizer, DistilBertModel
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        encoder = DistilBertModel.from_pretrained(args.model_name)
    model = MultiTaskDistilBert(encoder).to(device)

    collate = Collate(tokenizer, args.max_length)
    mk = lambda d, shuffle: DataLoader(d.to_dict("records"), batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)
    train_dl, val_dl, test_dl = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, total)))
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    step, done = 0, False
    for epoch in range(args.epochs):
        model.train()
        for b in train_dl:
            b = {k: v.to(device) for k, v in b.items()}
            logits, urg = model(b["input_ids"], b["attention_mask"])
            loss = ce(logits, b["labels"])
            mask = b["urgency"] >= 0
            if mask.any():
                loss = loss + args.urgency_weight * mse(urg[mask], b["urgency"][mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 20 == 0 or step == 1:
                print(f"epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        yv, pv, _, _ = predict(model, val_dl, device)
        print(f"epoch {epoch + 1} val accuracy {accuracy_score(yv, pv):.4f} (logged only; not used for selection)")
        if done:
            break

    # ---- single evaluation on held-out TEST split
    y, p, ug, up = predict(model, test_dl, device)
    metrics = {
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "test_accuracy": float(accuracy_score(y, p)),
        "test_macro_f1": float(f1_score(y, p, average="macro", labels=list(range(len(CATEGORIES))), zero_division=0)),
        "test_urgency_mae": urgency_mae(ug, up),
        "per_class": classification_report(y, p, labels=list(range(len(CATEGORIES))), target_names=CATEGORIES,
                                           output_dict=True, zero_division=0),
        "config": {k: v for k, v in vars(args).items()},
        "data_is_synthetic": synthetic,
        "optimizer_steps": step,
        "note": ("Trained on SYNTHETIC template data: not indicative of real-world performance." if synthetic else
                 "Computed on the held-out test split of the provided CSV."),
    }
    if args.tiny_offline or args.max_steps:
        metrics["note"] += " Smoke-test run (tiny/offline or truncated): metrics are meaningless."
    model.save(out_dir, tokenizer)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_png(y, p, out_dir / "confusion_matrix.png")
    print(f"TEST accuracy={metrics['test_accuracy']:.4f} macro_f1={metrics['test_macro_f1']:.4f}  saved to {out_dir}")


if __name__ == "__main__":
    main()
