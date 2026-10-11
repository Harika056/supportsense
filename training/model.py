"""Multi-task DistilBERT: shared encoder, a category head and an urgency head.

Design (deliberately simple): pooled [CLS] hidden state -> dropout ->
  * Linear(hidden, 8)  -> softmax category logits, trained with cross-entropy
  * Linear(hidden, 1)  -> sigmoid urgency in [0,1], trained with MSE
Total loss = CE + urgency_weight * MSE (urgency rows with no label are masked out).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from supportsense.labels import CATEGORIES

HEADS_FILE = "heads.pt"
LABELS_FILE = "labels.json"


class MultiTaskDistilBert(nn.Module):
    def __init__(self, encoder, labels: list[str] | None = None, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.labels = list(labels or CATEGORIES)
        hidden = encoder.config.dim
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, len(self.labels))
        self.urg_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.dropout(h)
        return self.cls_head(h), torch.sigmoid(self.urg_head(h)).squeeze(-1)

    def save(self, out_dir: str | Path, tokenizer) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out)
        tokenizer.save_pretrained(out)
        torch.save({"cls_head": self.cls_head.state_dict(), "urg_head": self.urg_head.state_dict()}, out / HEADS_FILE)
        (out / LABELS_FILE).write_text(json.dumps(self.labels))


def load_multitask(path: str):
    """Load a saved directory (encoder+tokenizer+heads) or a base HF model name."""
    from transformers import AutoTokenizer, DistilBertModel

    p = Path(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    encoder = DistilBertModel.from_pretrained(path)
    labels = None
    if p.is_dir() and (p / LABELS_FILE).exists():
        labels = json.loads((p / LABELS_FILE).read_text())
    model = MultiTaskDistilBert(encoder, labels)
    if p.is_dir() and (p / HEADS_FILE).exists():
        heads = torch.load(p / HEADS_FILE, map_location="cpu")
        model.cls_head.load_state_dict(heads["cls_head"])
        model.urg_head.load_state_dict(heads["urg_head"])
    return model, tokenizer
