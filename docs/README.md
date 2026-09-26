# Project Documentation

Architecture notes for the AI-powered fake news detection system with
blockchain-based source traceability.

## 1. High-level architecture

```
                 ┌──────────────────────────────────────────┐
  Browser ──────▶│  Flask (app.py)                          │
  API client ───▶│                                          │
                 │  /            /predict      /api/predict │
                 │  /history     /trace/<id>   /health       │
                 └───────┬───────────────┬───────────────────┘
                         │               │
              ┌──────────▼───────┐  ┌────▼──────────────────┐
              │ Model registry   │  │ Traceability service  │
              │ src/service.py   │  │ src/ledger.py         │
              │ (lazy load)      │  └────┬───────────┬───────┘
              └──────────┬───────┘       │           │
                         │        local hash     Ethereum
              ┌──────────▼───────┐  chain (SQL)   anchor (web3)
              │ NewsClassifier   │               │
              │ src/model.py     │        contracts/NewsTrace.sol
              └──────────┬───────┘
                         │
        ┌────────────────▼───────────────────┐
        │ TF-IDF (1-2 grams)                 │
        │ + style statistics                 │
        │ + Logistic Regression              │
        │ src/features.py / preprocessing.py │
        └────────────────────────────────────┘
```

## 2. Data flow for a single prediction

1. **Input validation** — minimum 40 characters, 2 MB request cap.
2. **Normalisation** — `clean_text()` strips HTML, URLs, entities, punctuation;
   lowercases and collapses whitespace; digits are preserved.
3. **Feature extraction**
   * `TfidfVectorizer(ngram_range=(1,2), max_features=20000, sublinear_tf=True)`
   * `StyleFeatureExtractor` → 8 dense columns: exclamation ratio, question
     ratio, ALL-CAPS ratio, average word length, type-token ratio, clickbait
     term density, quote ratio, digit ratio.
4. **Classification** — `LogisticRegression(class_weight="balanced")` returns
   `P(FAKE)` and `P(REAL)`.
5. **Persistence** — a `Prediction` row is written keyed by the SHA-256
   content hash, so identical submissions are de-duplicated.
6. **Traceability** — a `LedgerEntry` is appended (`prev_hash → entry_hash`),
   then optionally anchored on chain.
7. **Response** — redirect to `/result/<id>` (HTML) or `201` + JSON (API).

## 3. Ledger design

| Property | Choice | Why |
|---|---|---|
| Fingerprint | SHA-256 of *normalised* text | Same article with different whitespace maps to one hash |
| Integrity | `entry_hash = sha256(prediction_id ‖ content_hash ‖ prev_hash)` | Deletion or reordering of any entry breaks verification |
| On-chain payload | only the 32-byte digest | Flat gas cost, no copyrighted text stored publicly |
| Failure mode | local-only | A dead RPC endpoint must never fail a user request |

Verification lives at `GET /api/ledger/verify/<id>` and checks two things:

* `hash_matches` — does the stored article still hash to the stored digest?
* `chain_integrity` — does the whole hash chain still recompute?

## 4. Configuration

Read from environment / `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Flask session secret |
| `FLASK_DEBUG` | Dev server debug mode |
| `DATABASE_URL` | SQLAlchemy URI (defaults to SQLite) |
| `BLOCKCHAIN_RPC_URL` | Ethereum JSON-RPC endpoint |
| `BLOCKCHAIN_CONTRACT_ADDRESS` | Deployed `NewsTrace` address |
| `BLOCKCHAIN_PRIVATE_KEY` | Key used to send `record()` transactions |

## 5. Testing strategy

| File | Covers |
|---|---|
| `test_preprocessing.py` | cleaning, entity decoding, tokenisation, edge cases |
| `test_model.py` | style features, vectoriser, fit/predict/evaluate, save-load |
| `test_ledger.py` | content hashing, chain linking, tamper detection, service fallback |
| `test_app.py` | pages, API contract, dedup, ledger endpoints, form validation |

Run with `pytest -q`.

## 6. Known limitations & next steps

1. **Dataset bias** — the ISOT corpus is small and outlet-specific; numbers are
   optimistic. Retrain on LIAR / FakeNewsNet and report cross-source metrics.
2. **Explainability** — surface top TF-IDF contributing terms per prediction so
   the verdict is auditable, not just a probability.
3. **Source traceability depth** — capture the claimed publisher/URL and anchor
   that alongside the content hash to build a real provenance graph.
4. **Production serving** — add gunicorn/Waitress, a Postgres URI, structured
   logging and rate limiting on `/api/predict`.
