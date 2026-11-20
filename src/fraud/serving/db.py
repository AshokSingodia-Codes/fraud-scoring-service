import logging
import os
import threading
import time
from typing import Any

from sqlalchemy import Column, Float, Integer, String, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL_POOLED") or os.getenv("DATABASE_URL") or "sqlite:///./predictions.db"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

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
    notes = Column(String, nullable=True)


def init_db() -> None:
    """Initialize database tables within application lifespan."""
    Base.metadata.create_all(bind=engine)
    # Ensure backward-compatible column migration for SQLite
    try:
        with engine.connect() as conn:
            if engine.dialect.name == "sqlite":
                res = conn.exec_driver_sql("PRAGMA table_info(prediction_logs);").fetchall()
                col_names = [r[1] for r in res]
                if "notes" not in col_names:
                    conn.exec_driver_sql("ALTER TABLE prediction_logs ADD COLUMN notes VARCHAR;")
                    conn.commit()
    except Exception as e:
        logger.warning("Optional schema migration note: %s", e)


def get_db():
    """FastAPI dependency yielding a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure tables and migrations are initialized on module load
try:
    init_db()
except Exception as _e:
    pass


_db_lock = threading.Lock()
logger = logging.getLogger(__name__)


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
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
    except Exception as e:
        # Graceful degradation if database is unavailable
        logger.warning("Database log failed: %s", e)

