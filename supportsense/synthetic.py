"""Template-based SYNTHETIC support tickets.

These are generated from hand-written templates. They exist only so the pipeline
runs end-to-end with no external dataset. A model trained on them will look
unrealistically good on its own test split because the templates are easy and
repetitive: do not report such numbers as real-world performance.
"""
from __future__ import annotations

import random
from typing import Iterator

from .labels import CATEGORIES

_OPENERS = ["", "Hi, ", "Hello team, ", "Hey, ", "Good morning. ", "To whom it may concern, "]
_CLOSERS = ["", " Thanks.", " Please advise.", " Any help appreciated.", " Regards, Sam", " Thank you!"]
_PRODUCTS = ["Pro plan", "Starter plan", "mobile app", "web dashboard", "desktop client", "Team plan"]
_ITEMS = ["headphones", "standing desk", "laptop sleeve", "water bottle", "keyboard", "backpack"]

_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {product} subscription this month.",
        "My invoice shows the wrong amount, I was billed more than the listed price.",
        "Can you update the credit card on file? The payment keeps failing.",
        "There is an unexpected charge on my statement from your company.",
        "I need a copy of last month's invoice for the {product}.",
    ],
    "login_access": [
        "I can't log in to my account, it says my password is incorrect.",
        "The password reset email never arrives so I am locked out.",
        "Two-factor authentication code is not working and I cannot sign in.",
        "My account got locked after several login attempts.",
        "I forgot my username and cannot access the {product}.",
    ],
    "bug_report": [
        "The {product} crashes every time I open the settings page.",
        "I found a bug: the export button does nothing and throws an error.",
        "The page freezes when I upload a file larger than 10 MB.",
        "After the latest update the {product} shows a blank screen.",
        "Search returns an error message instead of results.",
    ],
    "feature_request": [
        "It would be great if you added a dark mode to the {product}.",
        "Please consider adding CSV export for reports.",
        "Feature request: support for single sign-on with our identity provider.",
        "Could you add the ability to schedule messages in advance?",
        "I wish the {product} had a keyboard shortcut for creating tickets.",
    ],
    "shipping_delivery": [
        "My order with the {item} has not arrived and tracking has not updated in a week.",
        "The package was marked as delivered but I never received the {item}.",
        "When will my {item} ship? The estimated delivery date has passed.",
        "The courier delivered my order to the wrong address.",
        "Can I change the shipping address for my pending order?",
    ],
    "refund_return": [
        "I want to return the {item} and get a refund.",
        "The {item} arrived damaged, please refund my purchase.",
        "How do I send back the {item} for a full refund?",
        "I requested a refund two weeks ago and have not received the money.",
        "Please issue a return label for the {item}, it does not fit.",
    ],
    "account_settings": [
        "How do I change the email address on my account?",
        "I want to update my profile name and phone number.",
        "Please help me turn off email notifications in my account settings.",
        "I need to delete my account and all associated data.",
        "How can I change my language and time zone preferences?",
    ],
    "general_inquiry": [
        "What are your customer support hours?",
        "Do you have a physical store or office I can visit?",
        "Can you tell me more about what the {product} includes?",
        "Who should I contact about a partnership opportunity?",
        "Is there a student discount available?",
    ],
}

_URGENT_PHRASES = [
    "This is urgent!", "I need this fixed immediately.", "This is a critical issue, ASAP please.",
    "Our whole team is blocked right now.", "Emergency: production is down.",
]
_CALM_PHRASES = ["No rush.", "Whenever you get a chance.", "Just curious.", "It is not urgent."]


def generate_ticket(rng: random.Random) -> dict:
    """One synthetic ticket: {text, label, urgency}. Urgency is a generated score in [0, 1]."""
    label = rng.choice(CATEGORIES)
    body = rng.choice(_TEMPLATES[label]).format(
        product=rng.choice(_PRODUCTS), item=rng.choice(_ITEMS)
    )
    roll = rng.random()
    if roll < 0.2:
        extra, urgency = rng.choice(_URGENT_PHRASES), rng.uniform(0.8, 1.0)
    elif roll < 0.45:
        extra, urgency = rng.choice(_CALM_PHRASES), rng.uniform(0.0, 0.2)
    else:
        extra, urgency = "", rng.uniform(0.3, 0.6)
    text = f"{rng.choice(_OPENERS)}{body} {extra}".strip() + rng.choice(_CLOSERS)
    return {"text": text, "label": label, "urgency": round(urgency, 3)}


def generate_tickets(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [generate_ticket(rng) for _ in range(n)]


def stream_tickets(seed: int | None = None) -> Iterator[dict]:
    rng = random.Random(seed)
    while True:
        yield generate_ticket(rng)
