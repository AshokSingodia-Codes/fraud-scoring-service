"""Pydantic v2 schemas for Fraud Scoring Service API (Stage 10)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """Input transaction payload schema."""

    model_config = ConfigDict(extra="allow")

    transaction_id: str = Field(..., description="Unique transaction ID")
    TransactionDT: int = Field(..., description="Time offset in seconds")
    TransactionAmt: float = Field(..., gt=0.0, description="Transaction amount in USD")
    ProductCD: str | None = Field(None, description="Product code")
    card1: int | None = Field(None, description="Card 1")
    card2: float | None = Field(None, description="Card 2")
    card3: float | None = Field(None, description="Card 3")
    card4: str | None = Field(None, description="Card 4")
    card5: float | None = Field(None, description="Card 5")
    card6: str | None = Field(None, description="Card 6")
    addr1: float | None = Field(None, description="Address 1")
    addr2: float | None = Field(None, description="Address 2")
    P_emaildomain: str | None = Field(None, description="Purchaser email domain")
    R_emaildomain: str | None = Field(None, description="Recipient email domain")
    DeviceType: str | None = Field(None, description="Device type")
    DeviceInfo: str | None = Field(None, description="Device info")


class ReasonItem(BaseModel):
    """Feature contribution reason."""

    feature: str
    label: str
    value: str | None = None
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    """Scoring response schema."""

    transaction_id: str
    fraud_probability: float
    decision: str  # approve, review, block
    risk_band: str  # Low, Medium, High
    reasons: list[ReasonItem] = []
    model_version: str
    thresholds: dict[str, float]
    latency_ms: float


class BatchScoreRequest(BaseModel):
    """Batch scoring request payload."""

    transactions: list[Transaction] = Field(..., max_length=1000)


class BatchScoreResponse(BaseModel):
    """Batch scoring response payload."""

    results: list[ScoreResponse]
    total_processed: int
    batch_latency_ms: float


class ModelInfoResponse(BaseModel):
    """Model metadata endpoint response."""

    model_version: str
    training_date: str
    thresholds: dict[str, float]
    num_features: int
    headline_metrics: dict[str, Any]
