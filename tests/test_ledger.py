"""Tests for the traceability ledger (real in-memory database)."""

import pytest

from src.ledger import GENESIS_HASH, LocalLedger, TraceabilityService, content_hash, get_service
from src.storage import LedgerEntry, db


# --------------------------------------------------------------------------- #
# content_hash
# --------------------------------------------------------------------------- #
def test_content_hash_is_stable_and_normalised():
    a = content_hash("The  Senate   passed the BILL.")
    b = content_hash("the senate passed the bill")
    assert a == b
    assert len(a) == 64


def test_content_hash_differs_for_different_text():
    assert content_hash("alpha") != content_hash("beta")


def test_content_hash_handles_empty_input():
    assert len(content_hash("")) == 64
    assert len(content_hash(None)) == 64


# --------------------------------------------------------------------------- #
# Hash chain
# --------------------------------------------------------------------------- #
def _make_entries(app):
    ledger = LocalLedger(db, LedgerEntry)
    for index in range(3):
        ledger.append(prediction_id=index + 1, hash_value=f"{index}" * 64)
    db.session.commit()
    return ledger, LedgerEntry.query.order_by(LedgerEntry.id).all()


def test_hash_chain_links_entries(db_app):
    with db_app.app_context():
        _ledger, entries = _make_entries(db_app)

        assert entries[0].prev_hash == GENESIS_HASH
        assert entries[1].prev_hash == entries[0].entry_hash
        assert entries[2].prev_hash == entries[1].entry_hash
        assert entries[1].entry_hash != entries[0].entry_hash


def test_verify_chain_accepts_intact_chain(db_app):
    with db_app.app_context():
        _ledger, entries = _make_entries(db_app)
        assert LocalLedger.verify_chain(entries) is True


def test_verify_chain_detects_content_tampering(db_app):
    with db_app.app_context():
        _ledger, entries = _make_entries(db_app)
        entries[0].content_hash = "f" * 64
        db.session.commit()
        assert LocalLedger.verify_chain(entries) is False


def test_verify_chain_detects_deleted_entry(db_app):
    with db_app.app_context():
        _ledger, entries = _make_entries(db_app)
        db.session.delete(entries[1])
        db.session.commit()
        remaining = LedgerEntry.query.order_by(LedgerEntry.id).all()
        # Removing a middle link breaks the prev-hash chain.
        assert LocalLedger.verify_chain(remaining) is False


def test_verify_chain_validates_a_segment(db_app):
    """A single record's entries form a segment that does not start at genesis."""
    with db_app.app_context():
        _ledger, entries = _make_entries(db_app)
        assert LocalLedger.verify_chain([entries[2]]) is True


def test_from_genesis_requires_the_first_link(db_app):
    with db_app.app_context():
        _ledger, entries = _make_entries(db_app)
        assert LocalLedger.verify_chain(entries, from_genesis=True) is True
        assert LocalLedger.verify_chain([entries[2]], from_genesis=True) is False


def test_all_entries_returns_the_whole_chain(db_app):
    with db_app.app_context():
        _ledger, entries = _make_entries(db_app)
        assert [e.id for e in LocalLedger.all_entries(LedgerEntry)] == [e.id for e in entries]


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
def test_service_falls_back_to_local_without_chain(db_app):
    with db_app.app_context():
        service = TraceabilityService(db, LedgerEntry, remote=None)

        assert service.chain_available is False
        status = service.status()
        assert status["local"]["available"] is True
        assert status["blockchain"]["available"] is False


def test_service_record_writes_local_entry(db_app):
    with db_app.app_context():
        service = TraceabilityService(db, LedgerEntry, remote=None)
        entry = service.record(prediction_id=1, hash_value="a" * 64)

        assert entry["network"] == "local-hashchain"
        assert entry["prev_hash"] == GENESIS_HASH
        assert len(entry["entry_hash"]) == 64


def test_get_service_without_rpc_config(db_app, monkeypatch):
    monkeypatch.delenv("BLOCKCHAIN_RPC_URL", raising=False)
    with db_app.app_context():
        service = get_service(db, LedgerEntry)
        assert service.remote is None
        assert service.chain_available is False


def test_get_service_with_unreachable_rpc(db_app, monkeypatch):
    monkeypatch.setenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("BLOCKCHAIN_CONTRACT_ADDRESS", "")
    with db_app.app_context():
        service = get_service(db, LedgerEntry)
        # Construction must never raise, even with a dead endpoint.
        assert service.remote is None or service.chain_available in (True, False)
