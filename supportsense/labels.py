"""Shared label set. Order defines class indices; do not reorder after training."""

CATEGORIES = [
    "billing",
    "login_access",
    "bug_report",
    "feature_request",
    "shipping_delivery",
    "refund_return",
    "account_settings",
    "general_inquiry",
]
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
URGENCY_THRESHOLD = 0.7  # tickets at/above this are flagged "urgent" in the UI
