# AI-Powered Fake News Detection and Source Traceability Using Blockchain

A complete Flask application that classifies news articles as **FAKE** or **REAL**
and records every verdict on an append-only, tamper-evident traceability ledger
(optionally anchored on an Ethereum compatible chain).

---

## Features

| Module | What it does |
|---|---|
| **NLP preprocessing** | `src/preprocessing.py` — HTML/URL/email stripping, entity decoding, normalisation, tokenisation with optional stopwords |
| **Feature engineering** | `src/features.py` — TF-IDF (unigrams + bigrams) combined with hand-crafted style statistics (ALL-CAPS ratio, exclamation spam, clickbait terms, type-token ratio) |
| **ML classifier** | `src/model.py` — Logistic Regression pipeline with `fit` / `predict` / `evaluate` / `save` / `load` |
| **Flask web app** | `app.py` — server rendered UI **and** a JSON API |
| **Database** | `src/storage.py` — Flask-SQLAlchemy models for predictions and ledger entries |
| **Blockchain traceability** | `src/ledger.py` + `contracts/NewsTrace.sol` — SHA-256 content fingerprints chained locally and anchored on chain when a node is available |
| **UI** | `templates/` + `static/` — check form, result, history, trace, model metrics |
| **Tests** | `tests/` — 52 pytest tests covering preprocessing, features, model, ledger and HTTP endpoints |

---

## Quick start

```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. Configuration
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux

# 3. Dataset (ISOT Fake/True corpus, ~1000 articles each)
python scripts/download_data.py

# 4. Train
python scripts/train.py

# 5. Run
python app.py                  # http://localhost:5000
```

### Verify

```bash
pytest -q          # 52 passed
curl http://localhost:5000/health
```

---

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI — paste an article |
| `POST` | `/predict` | Classify from the web form |
| `GET` | `/result/<id>` | Verdict page |
| `GET` | `/history` | Past classifications |
| `GET` | `/trace/<id>` | Hash-chain traceability view |
| `GET` | `/metrics` | Hold-out model metrics |
| `POST` | `/api/predict` | JSON: `{"text": "...", "title": "..."}` → `{result, record}` |
| `GET` | `/api/history` | JSON list of records (`?limit=`) |
| `GET` | `/api/predictions/<id>` | JSON record including full text |
| `GET` | `/api/ledger/status` | Local + Ethereum ledger status |
| `GET` | `/api/ledger/verify/<id>` | Verify hash match and chain integrity |
| `GET` | `/health` | Service health |

Example:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"WASHINGTON (Reuters) - The Senate passed the bill ...\"}"
```

Response:

```json
{
  "result": {
    "label": "REAL",
    "confidence": 0.9871,
    "is_fake": false,
    "probabilities": {"FAKE": 0.0129, "REAL": 0.9871}
  },
  "record": {"id": 1, "content_hash": "…", "predicted_label": "REAL"}
}
```

---

## How the traceability layer works

1. The article is normalised and hashed with **SHA-256** → `content_hash`.
2. A `LedgerEntry` is appended with `prev_hash` pointing at the previous entry,
   forming a **local hash chain**. Any edit or deletion breaks verification.
3. If `BLOCKCHAIN_RPC_URL` is reachable **and** `BLOCKCHAIN_CONTRACT_ADDRESS`
   is set, the same hash is anchored on chain via `NewsTrace.record()` and the
   transaction hash / block number are stored on the entry.
4. If the node is down the request still succeeds — the record simply stays
   `local-only`. Verification is available at `/api/ledger/verify/<id>`.

Deploy the contract with your preferred tooling (Hardhat / Foundry / Remix):

```bash
# contracts/NewsTrace.sol  — Solidity ^0.8.20
# then set in .env:
#   BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545
#   BLOCKCHAIN_CONTRACT_ADDRESS=0xYourContract
#   BLOCKCHAIN_PRIVATE_KEY=0x…   # funded testnet key
```

---

## Project layout

```
FakeNewsDetection/
├── app.py                  # Flask app (UI + JSON API)
├── config.py               # Env-driven configuration
├── requirements.txt
├── contracts/
│   └── NewsTrace.sol       # On-chain anchor contract
├── data/
│   ├── raw/                # Datasets (git-ignored)
│   └── processed/
├── docs/                   # Architecture & planning notes
├── models/                 # Trained artifacts + metrics.json
├── scripts/
│   ├── download_data.py    # Fetch a public dataset
│   └── train.py            # Train + evaluate + save
├── src/
│   ├── preprocessing.py    # NLP cleaning
│   ├── features.py         # TF-IDF + style features
│   ├── dataset.py          # CSV loading & splitting
│   ├── model.py            # Classifier
│   ├── ledger.py           # Hash chain + Ethereum anchor
│   ├── storage.py          # SQLAlchemy models
│   └── service.py          # Model registry (lazy loading)
├── static/                 # CSS + JS
├── templates/              # Jinja2 templates
└── tests/                  # pytest suite
```

---

## Honest notes on model quality

* The bundled dataset is the **ISOT** corpus (~1,991 usable articles). It is a
  benchmark-grade but *small* dataset, and real Reuters copy carries the
  `(Reuters)` marker, which the model legitimately leans on.
* The reported **99.5 %** hold-out accuracy is therefore optimistic. For a
  production claim you should retrain on a larger, multi-outlet corpus
  (LIAR, FakeNewsNet, GossipCop) and report cross-source evaluation.
* `scripts/download_data.py --dataset liar` fetches an alternative dataset —
  drop its CSVs into `data/raw/` and re-run `scripts/train.py`.
