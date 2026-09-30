"""Locust load testing script for Real-Time Fraud Scoring Service (Stage 11).

Simulates realistic traffic: 90% single /v1/score requests, 10% batch /v1/score/batch requests.
"""

import random
import uuid

from locust import HttpUser, between, task


def generate_synthetic_transaction(tx_id: str | None = None) -> dict:
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


class FraudScoringUser(HttpUser):
    wait_time = between(0.01, 0.05)
    headers = {
        "X-API-Key": "demo-api-key-12345",
        "Content-Type": "application/json",
    }

    @task(9)
    def score_single(self):
        """Single transaction scoring endpoint (90% traffic weight)."""
        payload = generate_synthetic_transaction()
        # Randomly toggle explain=true/false to test explanation overhead
        explain = random.choice([True, True, False])
        self.client.post(
            f"/v1/score?explain={str(explain).lower()}",
            headers=self.headers,
            json=payload,
            name="/v1/score",
        )

    @task(1)
    def score_batch(self):
        """Batch transaction scoring endpoint (10% traffic weight)."""
        batch_size = random.choice([10, 25, 50])
        payload = {
            "transactions": [
                generate_synthetic_transaction() for _ in range(batch_size)
            ]
        }
        self.client.post(
            "/v1/score/batch?explain=false",
            headers=self.headers,
            json=payload,
            name="/v1/score/batch",
        )
