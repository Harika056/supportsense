import pytest

from supportsense.classifier import KeywordClassifier, build_classifier
from supportsense.labels import CATEGORIES, ID2LABEL, LABEL2ID
from supportsense.synthetic import generate_tickets, stream_tickets
from streaming.router import CategoryRouter


def test_label_maps_are_consistent():
    assert len(CATEGORIES) == len(set(CATEGORIES)) == 8
    assert all(ID2LABEL[LABEL2ID[c]] == c for c in CATEGORIES)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I was charged twice on my invoice", "billing"),
        ("I can't log in, my password is wrong", "login_access"),
        ("The app crashes with an error", "bug_report"),
        ("Please consider adding dark mode, it would be great", "feature_request"),
        ("My package has not arrived, tracking is stuck", "shipping_delivery"),
        ("I want a refund, I will send back the item", "refund_return"),
        ("How do I change my profile notifications settings", "account_settings"),
        ("What are your store hours", "general_inquiry"),
    ],
)
def test_keyword_classifier_picks_obvious_categories(text, expected):
    pred = KeywordClassifier().predict(text)
    assert pred.label == expected
    assert 0 < pred.confidence <= 1
    assert set(pred.scores) == set(CATEGORIES)


def test_keyword_classifier_urgency_and_fallback():
    c = KeywordClassifier()
    assert c.predict("Our server is down, fix this immediately").urgency > 0.8
    assert c.predict("No rush, whenever you can").urgency < 0.2
    assert c.predict("zzz qqq").label == "general_inquiry"  # nothing matched -> fallback bucket


def test_predict_batch_keeps_order_and_length():
    texts = ["refund please", "crash error", "hours"]
    preds = KeywordClassifier().predict_batch(texts)
    assert [p.label for p in preds] == ["refund_return", "bug_report", "general_inquiry"]


def test_build_classifier_selects_backend():
    assert build_classifier("keyword").name == "keyword"
    with pytest.raises(ValueError):
        build_classifier("nope")


def test_synthetic_data_is_deterministic_and_valid():
    a, b = generate_tickets(50, seed=3), generate_tickets(50, seed=3)
    assert a == b
    assert a != generate_tickets(50, seed=4)
    assert all(t["label"] in CATEGORIES and 0 <= t["urgency"] <= 1 and t["text"] for t in a)
    assert {t["label"] for t in generate_tickets(400)} == set(CATEGORIES)


def test_stream_tickets_yields_forever():
    gen = stream_tickets(seed=1)
    assert len([next(gen) for _ in range(25)]) == 25


def test_router_routes_and_bounds_queues():
    r = CategoryRouter(maxlen=2)
    for i in range(3):
        r.route({"id": i, "label": "billing"})
    assert r.sizes()["billing"] == 2  # oldest dropped
    assert r.pop("billing")["id"] == 1
    assert r.pop("login_access") is None
    assert set(r.sizes()) == set(CATEGORIES)
