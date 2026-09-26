"""SQLAlchemy models and database helpers.

Kept separate from ``app.py`` so tests and scripts can create the schema
without importing the whole web application.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import String, Text

db = SQLAlchemy()


def utcnow() -> _dt.datetime:
    """Naive UTC timestamp (SQLite stores no timezone offset)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


class Prediction(db.Model):
    """A single classification result submitted through the app or API."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    content_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    title = db.Column(db.String(300), default="")
    text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    prob_fake = db.Column(db.Float, nullable=False, default=0.0)
    prob_real = db.Column(db.Float, nullable=False, default=0.0)
    model_version = db.Column(db.String(20), default="")
    source = db.Column(db.String(20), default="web")  # web | api
    created_at = db.Column(db.DateTime, default=utcnow, index=True)

    ledger_entries = db.relationship(
        "LedgerEntry",
        back_populates="prediction",
        cascade="all, delete-orphan",
        order_by="LedgerEntry.id",
    )

    def to_dict(self, include_text: bool = False) -> dict:
        payload = {
            "id": self.id,
            "content_hash": self.content_hash,
            "title": self.title,
            "predicted_label": self.predicted_label,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {"FAKE": round(self.prob_fake, 4), "REAL": round(self.prob_real, 4)},
            "model_version": self.model_version,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_text:
            payload["text"] = self.text
        return payload

    @property
    def tx_hash(self) -> Optional[str]:
        return self.ledger_entries[-1].tx_hash if self.ledger_entries else None


class LedgerEntry(db.Model):
    """Append-only traceability log for a prediction.

    Each entry stores the SHA-256 of the article plus, when an Ethereum
    compatible node is reachable, the transaction hash that anchored the
    content hash on chain. ``prev_hash`` links entries into a tamper
    evident hash chain even in fully offline mode.
    """

    __tablename__ = "ledger_entries"

    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=False, index=True)
    content_hash = db.Column(db.String(64), nullable=False, index=True)
    prev_hash = db.Column(db.String(64), nullable=False, default="0" * 64)
    entry_hash = db.Column(db.String(64), nullable=False)
    tx_hash = db.Column(db.String(80), nullable=True)
    block_number = db.Column(db.Integer, nullable=True)
    network = db.Column(db.String(40), nullable=False, default="local-hashchain")
    status = db.Column(db.String(20), nullable=False, default="recorded")
    created_at = db.Column(db.DateTime, default=utcnow)

    prediction = db.relationship("Prediction", back_populates="ledger_entries")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prediction_id": self.prediction_id,
            "content_hash": self.content_hash,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "network": self.network,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db(app) -> None:
    """Attach the DB to *app* and create tables if they do not exist."""
    if "sqlalchemy" not in app.extensions:
        db.init_app(app)
    with app.app_context():
        db.create_all()


def get_session_history(limit: int = 50) -> List[Prediction]:
    return (
        Prediction.query.order_by(Prediction.created_at.desc()).limit(limit).all()
    )


def find_by_hash(content_hash: str) -> Optional[Prediction]:
    return Prediction.query.filter_by(content_hash=content_hash).first()
