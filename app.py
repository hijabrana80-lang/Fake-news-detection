"""Flask application: web UI + JSON API for fake news detection.

Run it with::

    python app.py

or, with an auto-reloading dev server::

    flask --app app run --debug
"""

from __future__ import annotations

import json
import os

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

import config as app_config
from src.ledger import LocalLedger, content_hash, get_service
from src.service import METRICS_FILE, get_registry
from src.storage import LedgerEntry, Prediction, db, find_by_hash, get_session_history, init_db

MAX_ARTICLE_CHARS = 20_000
MIN_ARTICLE_CHARS = 40

registry = get_registry()


def create_app(config_name: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(app_config.get_config(config_name))

    init_db(app)
    app.ledger = get_service(db, LedgerEntry)

    _register_routes(app)
    _register_error_handlers(app)
    return app


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _register_routes(app: Flask) -> None:
    @app.get("/")
    def home():
        return render_template(
            "index.html",
            model_ready=registry.exists,
            model_error=registry.error,
            chain=app.ledger.status(),
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
            return redirect(url_for("home"))

        try:
            model = registry.get()
        except RuntimeError as exc:
            flash(str(exc), "error")
            return redirect(url_for("home"))

        result = model.predict_one(text[:MAX_ARTICLE_CHARS])
        record = _persist(model, title, text, result, source="web")

        return redirect(url_for("result", prediction_id=record.id))

    @app.get("/result/<int:prediction_id>")
    def result(prediction_id: int):
        record = db.session.get(Prediction, prediction_id) or abort(404)
        return render_template(
            "result.html",
            prediction=record,
            entries=record.ledger_entries,
            chain=app.ledger.status(),
        )

    @app.get("/history")
    def history():
        records = get_session_history(limit=100)
        return render_template("history.html", predictions=records)

    @app.get("/trace/<int:prediction_id>")
    def trace(prediction_id: int):
        record = db.session.get(Prediction, prediction_id) or abort(404)
        entries = record.ledger_entries
        return render_template(
            "trace.html",
            prediction=record,
            entries=entries,
            verified=LocalLedger.verify_chain(LocalLedger.all_entries(LedgerEntry), from_genesis=True),
            entries_ok=LocalLedger.verify_chain(entries),
            total_entries=len(LocalLedger.all_entries(LedgerEntry)),
            hash_ok=record.content_hash == content_hash(record.text),
            chain=app.ledger.status(),
        )

    @app.get("/metrics")
    def metrics():
        payload = {}
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, encoding="utf-8") as handle:
                payload = json.load(handle)
        return render_template("metrics.html", metrics=payload, model=registry.load())

    # ----------------------------------------------------------------- #
    # JSON API
    # ----------------------------------------------------------------- #
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
        return jsonify({"items": [r.to_dict() for r in get_session_history(limit)]})

    @app.get("/api/predictions/<int:prediction_id>")
    def api_prediction(prediction_id: int):
        record = db.session.get(Prediction, prediction_id) or abort(404)
        return jsonify(record.to_dict(include_text=True))

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

    app_ledger = getattr(_current_app(), "ledger", None)
    if app_ledger is not None:
        app_ledger.record(prediction_id=record.id, hash_value=digest)

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
        return redirect(url_for("home")), 413

    @app.errorhandler(500)
    def server_error(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal server error"}), 500
        return render_template("error.html", code=500, message="Something went wrong."), 500


app = create_app()


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "True").lower() in {"1", "true", "yes"}
    app.run(debug=debug, host="0.0.0.0", port=5000)
