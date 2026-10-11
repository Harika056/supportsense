import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from supportsense.classifier import TransformersClassifier
from supportsense.labels import CATEGORIES
from training import train as T


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    T.main(["--tiny-offline", "--num-synthetic", "240", "--epochs", "1", "--max-steps", "4",
            "--batch-size", "16", "--out-dir", str(out)])
    return out


def test_training_writes_all_artifacts(tiny_model_dir):
    names = {p.name for p in tiny_model_dir.iterdir()}
    assert {"heads.pt", "labels.json", "metrics.json", "confusion_matrix.png", "model.safetensors"} <= names
    assert json.loads((tiny_model_dir / "labels.json").read_text()) == CATEGORIES


def test_metrics_are_flagged_as_synthetic_and_smoke(tiny_model_dir):
    m = json.loads((tiny_model_dir / "metrics.json").read_text())
    assert m["data_is_synthetic"] is True
    assert "SYNTHETIC" in m["note"] and "meaningless" in m["note"]
    assert m["optimizer_steps"] == 4
    assert 0.0 <= m["test_accuracy"] <= 1.0
    assert set(m["split_sizes"]) == {"train", "val", "test"}


def test_saved_model_loads_and_predicts(tiny_model_dir):
    clf = TransformersClassifier(str(tiny_model_dir), max_length=32, device="cpu")
    preds = clf.predict_batch(["I was charged twice", "cannot log in", "dark mode please"])
    assert len(preds) == 3
    for p in preds:
        assert p.label in CATEGORIES
        assert abs(sum(p.scores.values()) - 1.0) < 0.01
        assert 0.0 <= p.urgency <= 1.0
    # batching must not change an individual prediction
    single = clf.predict("I was charged twice")
    assert single.label == preds[0].label
    assert single.confidence == pytest.approx(preds[0].confidence, abs=1e-3)


def args(**kw):
    ns = T.parse_args([])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_split_is_stratified_and_sized():
    df = pd.DataFrame(T.generate_tickets(800, seed=1))
    train, val, test = T.split_frame(df, 0.1, 0.1, seed=1)
    assert len(train) + len(val) + len(test) == 800
    assert abs(len(test) - 80) <= 2 and abs(len(val) - 80) <= 2
    assert set(test["label"]) == set(CATEGORIES)


def test_load_frame_rejects_bad_csv(tmp_path):
    missing = tmp_path / "a.csv"
    pd.DataFrame({"text": ["x"]}).to_csv(missing, index=False)
    with pytest.raises(SystemExit, match="missing required columns"):
        T.load_frame(args(data=str(missing)))
    unknown = tmp_path / "b.csv"
    pd.DataFrame({"text": ["x"], "label": ["spam"]}).to_csv(unknown, index=False)
    with pytest.raises(SystemExit, match="Unknown labels"):
        T.load_frame(args(data=str(unknown)))


def test_load_frame_accepts_csv_without_urgency(tmp_path):
    f = tmp_path / "c.csv"
    pd.DataFrame({"text": ["a", None], "label": ["billing", "billing"]}).to_csv(f, index=False)
    df, synthetic = T.load_frame(args(data=str(f)))
    assert synthetic is False and len(df) == 1 and df["urgency"].isna().all()


def test_urgency_mae_ignores_unlabelled_rows():
    import numpy as np

    assert T.urgency_mae(np.array([0.5, -1.0]), np.array([0.7, 0.0])) == pytest.approx(0.2)
    assert T.urgency_mae(np.array([-1.0]), np.array([0.3])) is None
