"""Database logging module using SQLAlchemy (Stage 10)."""

import os
import threading
import time
from typing import Any

from sqlalchemy import Column, Float, Integer, String, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL_POOLED") or os.getenv("DATABASE_URL") or "sqlite:///./predictions.db"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30.0} if DATABASE_URL.startswith("sqlite") else {},
)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    """Prediction audit log table."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, index=True)
    timestamp = Column(Float, default=time.time)
    fraud_probability = Column(Float)
    decision = Column(String)
    risk_band = Column(String)
    model_version = Column(String)
    latency_ms = Column(Float)


Base.metadata.create_all(bind=engine)

_db_lock = threading.Lock()


def log_prediction_to_db(score_res: dict[str, Any]) -> None:
    """Safely log prediction result to database."""
    try:
        with _db_lock:
            db = SessionLocal()
            try:
                log_entry = PredictionLog(
                    transaction_id=score_res["transaction_id"],
                    timestamp=time.time(),
                    fraud_probability=score_res["fraud_probability"],
                    decision=score_res["decision"],
                    risk_band=score_res["risk_band"],
                    model_version=score_res["model_version"],
                    latency_ms=score_res["latency_ms"],
                )
                db.add(log_entry)
                db.commit()
            finally:
                db.close()
    except Exception as e:
        # Graceful degradation if database is unavailable
        print(f"Warning: Database log failed: {e}")
