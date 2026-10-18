"""Synthetic transaction generation utility for testing and benchmarking."""

import random
import uuid
from typing import Any


def generate_synthetic_transaction(tx_id: str | None = None) -> dict[str, Any]:
    """Generate realistic synthetic transaction payload."""
    return {
        "transaction_id": tx_id or f"tx_{uuid.uuid4().hex[:12]}",
        "TransactionDT": random.randint(86400, 86400 * 180),
        "TransactionAmt": round(random.expovariate(1 / 100.0) + 1.0, 2),
        "ProductCD": random.choice(["W", "C", "R", "H", "S"]),
        "card1": random.randint(1000, 19999),
        "card2": float(random.randint(100, 600)),
        "card3": 150.0,
        "card4": random.choice(["visa", "mastercard", "discover", "american express"]),
        "card5": float(random.choice([102, 117, 126, 137, 166, 226])),
        "card6": random.choice(["debit", "credit"]),
        "addr1": float(random.randint(100, 500)),
        "addr2": 87.0,
        "P_emaildomain": random.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com"]),
        "R_emaildomain": random.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com"]),
        "DeviceType": random.choice(["desktop", "mobile"]),
        "DeviceInfo": random.choice(["Windows", "iOS Device", "MacOS", "Trident/7.0"]),
        "C1": float(random.randint(1, 20)),
        "C2": float(random.randint(1, 20)),
        "D1": float(random.randint(0, 300)),
    }
