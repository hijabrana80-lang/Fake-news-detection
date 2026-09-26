"""End-to-end tests for the Flask web app and JSON API."""

import json

import pytest

from src.ledger import content_hash
from src.storage import db

VALID_TEXT = (
    "WASHINGTON (Reuters) - The Senate approved a bipartisan infrastructure bill on Tuesday "
    "by a vote of 69-30, sending the measure to the House of Representatives, officials said. "
    "The package allocates new federal spending over five years for roads and bridges."
)

SHOUTY_TEXT = (
    "SHOCKING!!! You WON'T BELIEVE this one weird trick!! Doctors HATE it and big pharma is "
    "BANNING this miracle cure. Share before it is deleted forever, the truth they hide!!!"
)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"real" in response.data.lower()


def test_history_page_renders_empty(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert b"No classifications yet" in response.data


def test_metrics_page_renders(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_unknown_page_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "running"
    assert payload["model"]["available"] is True


def test_api_rejects_short_text(client):
    response = client.post("/api/predict", json={"text": "too short"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_api_rejects_missing_json(client):
    assert client.post("/api/predict", data="not json", content_type="text/plain").status_code == 400


def test_api_predicts_real_article(client):
    response = client.post("/api/predict", json={"text": VALID_TEXT, "title": "Senate vote"})
    assert response.status_code == 201

    payload = response.get_json()
    assert payload["result"]["label"] in {"FAKE", "REAL"}
    assert payload["result"]["is_fake"] == (payload["result"]["label"] == "FAKE")
    assert 0.0 <= payload["result"]["confidence"] <= 1.0
    assert payload["record"]["content_hash"] == content_hash(VALID_TEXT)


def test_api_predicts_fake_article(client):
    response = client.post("/api/predict", json={"text": SHOUTY_TEXT})
    assert response.status_code == 201
    assert response.get_json()["result"]["label"] == "FAKE"


def test_duplicate_text_reuses_the_same_record(client):
    first = client.post("/api/predict", json={"text": VALID_TEXT}).get_json()
    second = client.post("/api/predict", json={"text": VALID_TEXT}).get_json()

    assert first["record"]["id"] == second["record"]["id"]
    assert len(client.get("/api/history").get_json()["items"]) == 1


def test_api_history_lists_records(client):
    client.post("/api/predict", json={"text": SHOUTY_TEXT})
    items = client.get("/api/history").get_json()["items"]
    assert len(items) == 1
    assert items[0]["predicted_label"] == "FAKE"


def test_api_fetches_single_prediction(client):
    created = client.post("/api/predict", json={"text": VALID_TEXT}).get_json()
    response = client.get(f"/api/predictions/{created['record']['id']}")
    assert response.status_code == 200
    assert response.get_json()["text"] == VALID_TEXT


def test_api_missing_prediction_is_404(client):
    assert client.get("/api/predictions/9999").status_code == 404


# --------------------------------------------------------------------------- #
# Ledger / traceability
# --------------------------------------------------------------------------- #
def test_prediction_is_written_to_the_ledger(client):
    created = client.post("/api/predict", json={"text": VALID_TEXT}).get_json()
    prediction_id = created["record"]["id"]

    trace = client.get(f"/trace/{prediction_id}")
    assert trace.status_code == 200
    assert b"Full ledger verified" in trace.data

    verify = client.get(f"/api/ledger/verify/{prediction_id}").get_json()
    assert verify["hash_matches"] is True
    assert verify["record_entries_valid"] is True
    assert verify["chain_integrity"] is True
    assert len(verify["entries"]) == 1
    assert verify["entries"][0]["prev_hash"] == "0" * 64


def test_chain_integrity_holds_across_multiple_records(client):
    first = client.post("/api/predict", json={"text": SHOUTY_TEXT}).get_json()
    second = client.post("/api/predict", json={"text": VALID_TEXT}).get_json()

    for created in (first, second):
        verify = client.get(f"/api/ledger/verify/{created['record']['id']}").get_json()
        assert verify["chain_integrity"] is True
        assert verify["record_entries_valid"] is True

    verify = client.get(f"/api/ledger/verify/{second['record']['id']}").get_json()
    assert verify["total_ledger_entries"] == 2

    trace = client.get(f"/trace/{second['record']['id']}")
    assert b"Full ledger verified" in trace.data


def test_chain_integrity_detects_tampering(client, app):
    created = client.post("/api/predict", json={"text": VALID_TEXT}).get_json()
    prediction_id = created["record"]["id"]

    with app.app_context():
        from src.storage import LedgerEntry

        entry = LedgerEntry.query.first()
        entry.content_hash = "f" * 64
        db.session.commit()

    verify = client.get(f"/api/ledger/verify/{prediction_id}").get_json()
    assert verify["chain_integrity"] is False
    assert verify["record_entries_valid"] is False

    trace = client.get(f"/trace/{prediction_id}")
    assert b"verification FAILED" in trace.data


def test_ledger_status_endpoint(client):
    payload = client.get("/api/ledger/status").get_json()
    assert payload["local"]["available"] is True
    assert "blockchain" in payload


def test_result_page_renders(client):
    created = client.post("/api/predict", json={"text": SHOUTY_TEXT}).get_json()
    response = client.get(f"/result/{created['record']['id']}")
    assert response.status_code == 200
    assert b"confidence" in response.data


# --------------------------------------------------------------------------- #
# Web form
# --------------------------------------------------------------------------- #
def test_form_rejects_short_text(client):
    response = client.post("/predict", data={"text": "hi"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"too short" in response.data


def test_form_predicts_and_redirects(client):
    response = client.post(
        "/predict",
        data={"title": "Senate", "text": SHOUTY_TEXT},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/result/" in response.headers["Location"]


def test_form_without_model_shows_error(db_app, monkeypatch):
    from app import registry

    monkeypatch.setattr(registry, "path", "/nonexistent/model.joblib")
    registry.unload()

    response = db_app.test_client().post(
        "/predict", data={"text": SHOUTY_TEXT}, follow_redirects=True
    )
    assert response.status_code == 200
    assert b"not trained" in response.data or b"Model file not found" in response.data

    monkeypatch.undo()


# --------------------------------------------------------------------------- #
# Payload size
# --------------------------------------------------------------------------- #
def test_oversized_payload_is_rejected(client):
    payload = json.dumps({"text": "a" * (2 * 1024 * 1024 + 100)})
    response = client.post(
        "/api/predict", data=payload, content_type="application/json"
    )
    assert response.status_code in (413, 400)


@pytest.mark.parametrize("endpoint", ["/api/ledger/status", "/health"])
def test_api_endpoints_return_json(client, endpoint):
    response = client.get(endpoint)
    assert response.content_type.startswith("application/json")
    assert response.get_json() is not None
