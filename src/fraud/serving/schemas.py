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
    velocity_features: dict[str, float] = {}
    shadow_evaluation: dict[str, Any] | None = None
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


class PredictionRecordResponse(BaseModel):
    """Prediction audit log record schema."""

    transaction_id: str
    timestamp: float
    fraud_probability: float
    decision: str
    risk_band: str
    model_version: str
    latency_ms: float
    notes: str | None = None


class PredictionListResponse(BaseModel):
    """Paginated list of prediction audit log records."""

    total: int
    limit: int
    offset: int
    items: list[PredictionRecordResponse]


class PredictionCreateRequest(BaseModel):
    """Request payload for manual audit record creation."""

    transaction_id: str = Field(..., description="Unique transaction ID")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability score")
    decision: str = Field(..., description="Decision: approve, review, or block")
    risk_band: str = Field("Medium", description="Risk band: Low, Medium, or High")
    model_version: str = Field("1.0.0", description="Model version string")
    latency_ms: float = Field(0.0, ge=0.0, description="Execution latency in ms")
    notes: str | None = Field(None, description="Optional analyst notes")


class PredictionUpdateRequest(BaseModel):
    """Request payload for updating an audit record."""

    decision: str | None = Field(None, description="Updated decision: approve, review, or block")
    notes: str | None = Field(None, description="Updated analyst notes")
