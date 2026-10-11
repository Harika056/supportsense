import pytest

pd = pytest.importorskip("pandas")

from scripts import prepare_csv as P


def raw():
    return pd.DataFrame({
        "desc": ["Charged twice ", "App crashes", "Charged twice ", "Hello?", ""],
        "kind": ["Billing inquiry", "Technical issue", "Billing inquiry", "Other", "Billing inquiry"],
        "prio": [0.9, 2.0, 0.1, None, 0.5],
    })


MAP = {"Billing inquiry": "billing", "Technical issue": "bug_report"}


def test_prepare_maps_labels_drops_unmapped_duplicates_and_blank():
    out = P.prepare(raw(), "desc", "kind", MAP)
    assert list(out["text"]) == ["Charged twice", "App crashes"]
    assert list(out["label"]) == ["billing", "bug_report"]


def test_prepare_clips_urgency_between_zero_and_one():
    out = P.prepare(raw(), "desc", "kind", MAP, urgency_col="prio")
    assert list(out["urgency"]) == [0.9, 1.0]


def test_prepare_rejects_missing_column_and_unknown_target():
    with pytest.raises(ValueError, match="not found"):
        P.prepare(raw(), "nope", "kind", MAP)
    with pytest.raises(ValueError, match="unknown target"):
        P.prepare(raw(), "desc", "kind", {"Billing inquiry": "spam"})


def test_parse_map():
    assert P.parse_map(["A b=billing", " X = bug_report "]) == {"A b": "billing", "X": "bug_report"}
    with pytest.raises(ValueError):
        P.parse_map(["no-equals"])


def test_cli_roundtrip(tmp_path, capsys):
    src, dst = tmp_path / "raw.csv", tmp_path / "out" / "clean.csv"
    raw().to_csv(src, index=False)
    P.main(["--input", str(src), "--text-col", "desc", "--label-col", "kind",
            "--map", "Billing inquiry=billing", "--map", "Technical issue=bug_report", "--out", str(dst)])
    assert len(pd.read_csv(dst)) == 2
    P.main(["--input", str(src), "--text-col", "desc", "--label-col", "kind", "--list-labels"])
    assert "Billing inquiry" in capsys.readouterr().out
