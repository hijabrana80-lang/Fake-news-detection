"""Flask application: web UI + JSON API for fake news detection.

Run it with::

    python app.py

or, with an auto-reloading dev server::

    flask --app app run --debug
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy import func

import config as app_config
from src.dataset import FAKE, REAL
from src.ledger import LocalLedger, content_hash, get_service
from src.service import METRICS_FILE, get_registry
from src.storage import LedgerEntry, Prediction, db, find_by_hash, get_session_history, init_db

MAX_ARTICLE_CHARS = 20_000
MIN_ARTICLE_CHARS = 40

registry = get_registry()

_REF_RE = re.compile(r"^news-?0*(\d+)$", re.IGNORECASE)
_FULL_HASH_RE = re.compile(r"^(?:0x)?[0-9a-f]{64}$", re.IGNORECASE)
_HASH_PREFIX_RE = re.compile(r"^(?:0x)?[0-9a-f]{8,}$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def record_ref(prediction_id: int) -> str:
    """Human friendly public identifier, e.g. ``NEWS-00042``."""
    return f"NEWS-{prediction_id:05d}"


def source_status(record: Prediction) -> dict:
    """Describe how (and where) this record is anchored.

    Deliberately distinguishes an *on-chain* anchor from the local integrity
    ledger so the UI can never claim "blockchain verified" without proof.
    """
    entries = record.ledger_entries
    if not entries:
        return {"code": "none", "label": "No record", "on_chain": False}

    entry = entries[-1]
    if entry.tx_hash and entry.status == "confirmed":
        return {"code": "onchain", "label": "Blockchain record", "on_chain": True}
    if entry.tx_hash:
        return {"code": "pending", "label": "Chain transaction pending", "on_chain": False}
    return {"code": "local", "label": "Local integrity record", "on_chain": False}


def _verification_payload(record: Prediction, query: str, ledger) -> dict:
    """Verification response shared by the POST and GET verify endpoints."""
    hash_ok = record.content_hash == content_hash(record.text)
    chain_ok = LocalLedger.verify_chain(
        LocalLedger.all_entries(LedgerEntry), from_genesis=True
    )
    status = source_status(record)
    entry = record.ledger_entries[-1] if record.ledger_entries else None

    return {
        "found": True,
        "query": query,
        "ref": record_ref(record.id),
        "integrity": "VERIFIED" if (hash_ok and chain_ok) else "FAILED",
        "hash_matches": hash_ok,
        "chain_integrity": chain_ok,
        "record": record.to_dict(),
        "source_status": status,
        "blockchain": {
            "available": bool(ledger.chain_available and status["on_chain"]),
            "network": entry.network if entry else None,
            "tx_hash": entry.tx_hash if entry else None,
            "block_number": entry.block_number if entry else None,
        },
        "entries": [e.to_dict() for e in record.ledger_entries],
    }


def load_metrics() -> dict:
    """Read the saved hold-out evaluation, if the model has been trained."""
    if not os.path.exists(METRICS_FILE):
        return {}
    try:
        with open(METRICS_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _stats_payload(limit: int = 8) -> dict:
    total = db.session.query(func.count(Prediction.id)).scalar() or 0
    fake = (
        db.session.query(func.count(Prediction.id)).filter(Prediction.predicted_label == FAKE).scalar() or 0
    )
    real = (
        db.session.query(func.count(Prediction.id)).filter(Prediction.predicted_label == REAL).scalar() or 0
    )
    on_chain = (
        db.session.query(func.count(LedgerEntry.id)).filter(LedgerEntry.tx_hash.isnot(None)).scalar() or 0
    )
    entries_total = db.session.query(func.count(LedgerEntry.id)).scalar() or 0
    avg_confidence = db.session.query(func.avg(Prediction.confidence)).scalar() or 0.0

    recent = get_session_history(limit=limit)
    return {
        "total": int(total),
        "real": int(real),
        "fake": int(fake),
        "on_chain": int(on_chain),
        "ledger_entries": int(entries_total),
        "local_records": int(total) - int(on_chain),
        "avg_confidence": round(float(avg_confidence), 4),
        "blockchain_available": False,
        "recent": [
            {
                "id": record_ref(r.id),
                "numeric_id": r.id,
                "headline": r.title or (r.text[:90] + ("…" if len(r.text) > 90 else "")),
                "prediction": r.predicted_label,
                "confidence": round(float(r.confidence), 4),
                "source": source_status(r),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recent
        ],
    }


def _resolve_record(query: str):
    """Look a record up by public ID, numeric ID, content hash or tx hash."""
    q = (query or "").strip()
    if not q:
        return None

    match = _REF_RE.match(q)
    if match:
        return db.session.get(Prediction, int(match.group(1)))

    if q.isdigit():
        return db.session.get(Prediction, int(q))

    lowered = q.lower()
    if _FULL_HASH_RE.match(q):
        digest = lowered[2:] if lowered.startswith("0x") else lowered
        record = find_by_hash(digest)
        if record is not None:
            return record
        entry = (
            LedgerEntry.query.filter(
                (func.lower(LedgerEntry.tx_hash) == lowered)
                | (func.lower(LedgerEntry.entry_hash) == digest)
            )
            .first()
        )
        if entry is not None:
            return entry.prediction
        return None

    if _HASH_PREFIX_RE.match(q):
        entry = (
            LedgerEntry.query.filter(
                func.lower(LedgerEntry.tx_hash).like(f"{lowered}%")
                | func.lower(LedgerEntry.content_hash).like(f"{lowered}%")
                | func.lower(LedgerEntry.entry_hash).like(f"{lowered}%")
            )
            .first()
        )
        if entry is not None:
            return entry.prediction

        record = Prediction.query.filter(Prediction.content_hash.like(f"{lowered}%")).first()
        return record

    return None


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(config_name: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(app_config.get_config(config_name))

    init_db(app)
    app.ledger = get_service(db, LedgerEntry)

    _register_routes(app)
    _register_error_handlers(app)
    return app


def _register_routes(app: Flask) -> None:
    @app.context_processor
    def inject_globals():
        return {"year": datetime.now(timezone.utc).year, "brand": "TRUTHLINE"}

    # ----------------------------------------------------------------- #
    # Pages
    # ----------------------------------------------------------------- #
    @app.get("/")
    def home():
        return render_template(
            "home.html",
            recent=get_session_history(limit=3),
            accuracy=load_metrics().get("accuracy"),
            model_ready=registry.exists,
            chain=app.ledger.status(),
            page="home",
        )

    @app.get("/analyze")
    def analyze():
        return render_template(
            "analyze.html",
            model_ready=registry.exists,
            model_error=registry.error,
            chain=app.ledger.status(),
            page="analyze",
        )

    @app.post("/predict")
    def predict():
        title = (request.form.get("title") or "").strip()[:300]
        text = (request.form.get("text") or "").strip()

        if len(text) < MIN_ARTICLE_CHARS:
            flash(
                f"Article text is too short — at least {MIN_ARTICLE_CHARS} characters are required.",
                "error",
            )
            return redirect(url_for("analyze"))

        try:
            model = registry.get()
        except RuntimeError as exc:
            flash(str(exc), "error")
            return redirect(url_for("analyze"))

        result = model.predict_one(text[:MAX_ARTICLE_CHARS])
        record = _persist(model, title, text, result, source="web")
        return redirect(url_for("result", prediction_id=record.id))

    @app.get("/result/<int:prediction_id>")
    def result(prediction_id: int):
        record = db.session.get(Prediction, prediction_id) or abort(404)
        return render_template(
            "result.html",
            prediction=record,
            ref=record_ref(record.id),
            entries=record.ledger_entries,
            source=source_status(record),
            chain=app.ledger.status(),
            page="analyze",
        )

    @app.get("/verify")
    def verify():
        query = (request.args.get("q") or "").strip()
        found = None
        resolution = None

        if query:
            found = _resolve_record(query)
            if found is None:
                resolution = {"found": False, "query": query}
            else:
                payload = _verification_payload(found, query, app.ledger)
                resolution = {
                    **payload,
                    "record": found,
                    "entries": found.ledger_entries,
                    "on_chain": payload["source_status"]["on_chain"],
                }

        return render_template(
            "verify.html",
            query=query,
            resolution=resolution,
            chain=app.ledger.status(),
            page="verify",
        )

    @app.get("/dashboard")
    def dashboard():
        stats = _stats_payload()
        stats["blockchain_available"] = app.ledger.chain_available
        return render_template(
            "dashboard.html",
            stats=stats,
            chain=app.ledger.status(),
            page="dashboard",
        )

    @app.get("/how-it-works")
    def how_it_works():
        return render_template("how_it_works.html", page="how")

    @app.get("/about")
    def about():
        return render_template("about.html", page="about")

    @app.get("/history")
    def history():
        records = get_session_history(limit=100)
        return render_template("history.html", predictions=records, page="dashboard")

    @app.get("/trace/<int:prediction_id>")
    def trace(prediction_id: int):
        record = db.session.get(Prediction, prediction_id) or abort(404)
        entries = record.ledger_entries
        return render_template(
            "trace.html",
            prediction=record,
            ref=record_ref(record.id),
            entries=entries,
            verified=LocalLedger.verify_chain(
                LocalLedger.all_entries(LedgerEntry), from_genesis=True
            ),
            entries_ok=LocalLedger.verify_chain(entries),
            total_entries=len(LocalLedger.all_entries(LedgerEntry)),
            hash_ok=record.content_hash == content_hash(record.text),
            source=source_status(record),
            chain=app.ledger.status(),
            page="verify",
        )

    @app.get("/metrics")
    def metrics():
        return render_template(
            "metrics.html",
            metrics=load_metrics(),
            model=registry.load(),
            page="dashboard",
        )

    # ----------------------------------------------------------------- #
    # JSON API
    # ----------------------------------------------------------------- #
    @app.get("/api/stats")
    def api_stats():
        stats = _stats_payload(limit=12)
        stats["blockchain_available"] = app.ledger.chain_available
        stats["ledger"] = app.ledger.status()
        return jsonify(stats)

    @app.post("/api/verify")
    def api_verify():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "'query' is required"}), 400

        record = _resolve_record(query)
        if record is None:
            return jsonify({"found": False, "query": query}), 404
        return jsonify(_verification_payload(record, query, app.ledger))

    @app.get("/api/verify/<path:ref>")
    def api_verify_get(ref: str):
        record = _resolve_record(ref)
        if record is None:
            return jsonify({"found": False, "query": ref}), 404
        return jsonify(_verification_payload(record, ref, app.ledger))

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        title = (payload.get("title") or "").strip()[:300]

        if len(text) < MIN_ARTICLE_CHARS:
            return jsonify({"error": f"'text' must be at least {MIN_ARTICLE_CHARS} characters"}), 400

        try:
            model = registry.get()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

        result = model.predict_one(text[:MAX_ARTICLE_CHARS])
        record = _persist(model, title, text, result, source="api")

        return jsonify({"result": result, "record": record.to_dict()}), 201

    @app.get("/api/history")
    def api_history():
        limit = min(int(request.args.get("limit", 50)), 500)
        return jsonify(
            {
                "items": [
                    {**r.to_dict(), "ref": record_ref(r.id), "source_status": source_status(r)}
                    for r in get_session_history(limit)
                ]
            }
        )

    @app.get("/api/predictions/<int:prediction_id>")
    def api_prediction(prediction_id: int):
        record = db.session.get(Prediction, prediction_id) or abort(404)
        payload = record.to_dict(include_text=True)
        payload["ref"] = record_ref(record.id)
        payload["source_status"] = source_status(record)
        return jsonify(payload)

    @app.get("/api/ledger/status")
    def api_ledger_status():
        return jsonify(app.ledger.status())

    @app.get("/api/ledger/verify/<int:prediction_id>")
    def api_ledger_verify(prediction_id: int):
        record = db.session.get(Prediction, prediction_id) or abort(404)
        all_entries = LocalLedger.all_entries(LedgerEntry)
        return jsonify(
            {
                "prediction_id": record.id,
                "ref": record_ref(record.id),
                "content_hash": record.content_hash,
                "hash_matches": record.content_hash == content_hash(record.text),
                "record_entries_valid": LocalLedger.verify_chain(record.ledger_entries),
                "chain_integrity": LocalLedger.verify_chain(all_entries, from_genesis=True),
                "total_ledger_entries": len(all_entries),
                "entries": [e.to_dict() for e in record.ledger_entries],
            }
        )

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "running",
                "message": "Application is running successfully",
                "model": {
                    "loaded": registry.is_loaded,
                    "available": registry.exists,
                    "error": registry.error,
                },
                "database": {"connected": db.engine is not None},
                "ledger": app.ledger.status(),
            }
        )


def _persist(model, title: str, text: str, result: dict, *, source: str) -> Prediction:
    """Store the prediction and anchor it in the traceability ledger."""
    digest = content_hash(text)
    existing = find_by_hash(digest)
    if existing is not None:
        return existing

    probs = result.get("probabilities", {})
    record = Prediction(
        content_hash=digest,
        title=title,
        text=text,
        predicted_label=result["label"],
        confidence=result["confidence"],
        prob_fake=float(probs.get("FAKE", 0.0)),
        prob_real=float(probs.get("REAL", 0.0)),
        model_version=model.summary().get("model_version", ""),
        source=source,
    )
    db.session.add(record)
    db.session.flush()

    ledger = getattr(_current_app(), "ledger", None)
    if ledger is not None:
        ledger.record(prediction_id=record.id, hash_value=digest)

    db.session.commit()
    return record


def _current_app():
    from flask import current_app

    return current_app


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(_error):
        message = "The submitted article is too large (2 MB max)."
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), 413
        flash(message, "error")
        return redirect(url_for("analyze")), 413

    @app.errorhandler(500)
    def server_error(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal server error"}), 500
        return render_template("error.html", code=500, message="Something went wrong."), 500


app = create_app()


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "True").lower() in {"1", "true", "yes"}
    app.run(debug=debug, host="0.0.0.0", port=5000)
