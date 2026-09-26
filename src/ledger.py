"""Source traceability layer.

Two ledgers are used together:

``LocalLedger``
    A tamper-evident hash chain persisted in the database. Always available,
    needs no infrastructure, and lets a user verify that a record has not
    been altered.

``EthereumLedger``
    Anchors the same content hash on an Ethereum compatible chain through a
    deployed ``NewsTrace`` contract (see ``contracts/NewsTrace.sol``).
    Used automatically when ``BLOCKCHAIN_RPC_URL`` is reachable and a
    contract address is configured.

The service never fails a prediction because the chain is down: the local
entry is always written, and the on-chain transaction is attached later if
possible.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

__all__ = ["content_hash", "LocalLedger", "EthereumLedger", "TraceabilityService", "get_service"]

GENESIS_HASH = "0" * 64

# Minimal ABI for contracts/NewsTrace.sol — only what we call.
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"internalType": "string", "name": "sourceUrl", "type": "string"},
        ],
        "name": "record",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "contentHash", "type": "bytes32"}],
        "name": "verify",
        "outputs": [
            {"internalType": "bool", "name": "found", "type": "bool"},
            {"internalType": "address", "name": "reporter", "type": "address"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "string", "name": "sourceUrl", "type": "string"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "contentHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "reporter", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": False, "internalType": "string", "name": "sourceUrl", "type": "string"},
        ],
        "name": "NewsRecorded",
        "type": "event",
    },
]


def content_hash(text: str) -> str:
    """SHA-256 of the normalised article text.

    Normalisation (lowercase, punctuation stripped, whitespace collapsed) is
    shared with the ML preprocessing pipeline so that re-submitting the same
    article with trivial formatting differences yields the same fingerprint.
    """
    from .preprocessing import clean_text

    return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()


class LocalLedger:
    """Hash-chain backed store, written straight into the app database."""

    network = "local-hashchain"

    def __init__(self, db, model):
        self.db = db
        self.Entry = model

    def last_hash(self) -> str:
        last = self.Entry.query.order_by(self.Entry.id.desc()).first()
        return last.entry_hash if last else GENESIS_HASH

    def append(self, *, prediction_id: int, hash_value: str, status: str = "recorded") -> dict:
        prev_hash = self.last_hash()
        stamp = f"{prediction_id}:{hash_value}:{prev_hash}"
        entry_hash = hashlib.sha256(stamp.encode("utf-8")).hexdigest()

        entry = self.Entry(
            prediction_id=prediction_id,
            content_hash=hash_value,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            network=self.network,
            status=status,
        )
        self.db.session.add(entry)
        self.db.session.flush()
        return entry.to_dict()

    @staticmethod
    def verify_chain(entries, *, from_genesis: bool = False) -> bool:
        """Recompute the chain to detect any tampering.

        ``entries`` must be contiguous, oldest first. By default a segment is
        accepted (needed when validating a single record's entries); pass
        ``from_genesis=True`` for a whole-ledger audit, which additionally
        requires the first entry to start at the genesis hash.
        """
        entries = list(entries)
        if from_genesis and entries and entries[0].prev_hash != GENESIS_HASH:
            return False

        prev = None
        for entry in entries:
            if prev is not None and entry.prev_hash != prev:
                return False
            expected = hashlib.sha256(
                f"{entry.prediction_id}:{entry.content_hash}:{entry.prev_hash}".encode("utf-8")
            ).hexdigest()
            if entry.entry_hash != expected:
                return False
            prev = entry.entry_hash
        return True

    @staticmethod
    def all_entries(entry_model):
        return entry_model.query.order_by(entry_model.id.asc()).all()


class EthereumLedger:
    """Anchors content hashes on chain through the ``NewsTrace`` contract."""

    def __init__(self, rpc_url: str, contract_address: str, private_key: str | None = None):
        from web3 import Web3

        self.rpc_url = rpc_url
        self.private_key = private_key or os.getenv("BLOCKCHAIN_PRIVATE_KEY", "")
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 5}))
        self.address = Web3.to_checksum_address(contract_address) if contract_address else None
        self.contract = (
            self.w3.eth.contract(address=self.address, abi=CONTRACT_ABI) if self.address else None
        )

    @property
    def available(self) -> bool:
        try:
            return bool(self.w3.is_connected() and self.contract is not None)
        except Exception:
            return False

    @property
    def network(self) -> str:
        try:
            return f"evm:{self.w3.eth.chain_id}"
        except Exception:
            return "evm:unknown"

    @property
    def account(self) -> Optional[str]:
        if not self.private_key:
            return None
        try:
            from web3 import Web3

            return Web3.to_checksum_address(self.w3.eth.account.from_key(self.private_key).address)
        except Exception:
            return None

    def record(self, hash_value: str, source_url: str = "") -> dict:
        """Send ``record(contentHash, sourceUrl)`` and wait for the receipt."""
        if not self.available:
            raise RuntimeError("Ethereum node or contract not configured")
        if not self.account:
            raise RuntimeError("BLOCKCHAIN_PRIVATE_KEY is not set")

        hash_bytes = bytes.fromhex(hash_value)
        tx = self.contract.functions.record(hash_bytes, source_url).build_transaction(
            {
                "from": self.account,
                "nonce": self.w3.eth.get_transaction_count(self.account),
                "gas": 200_000,
            }
        )
        signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        return {
            "tx_hash": tx_hash.hex(),
            "block_number": int(receipt.blockNumber),
            "status": "confirmed" if receipt.status == 1 else "reverted",
            "network": self.network,
        }

    def verify(self, hash_value: str) -> dict:
        if not self.available:
            return {"found": False, "reason": "chain unavailable"}
        found, reporter, timestamp, source_url = self.contract.functions.verify(
            bytes.fromhex(hash_value)
        ).call()
        return {
            "found": bool(found),
            "reporter": reporter,
            "timestamp": int(timestamp),
            "source_url": source_url,
        }

    def status(self) -> dict:
        try:
            connected = self.w3.is_connected()
            return {
                "rpc_url": self.rpc_url,
                "connected": connected,
                "contract": self.address,
                "chain_id": self.w3.eth.chain_id if connected else None,
                "account": self.account,
            }
        except Exception as exc:  # pragma: no cover - network dependent
            return {"rpc_url": self.rpc_url, "connected": False, "error": str(exc)}


class TraceabilityService:
    """Writes the local hash chain and, when possible, the on-chain anchor."""

    def __init__(self, db, entry_model, remote: Optional[EthereumLedger] = None):
        self.db = db
        self.local = LocalLedger(db, entry_model)
        self.remote = remote

    @property
    def chain_available(self) -> bool:
        return bool(self.remote and self.remote.available)

    def record(self, *, prediction_id: int, hash_value: str, source_url: str = "") -> dict:
        entry = self.local.append(prediction_id=prediction_id, hash_value=hash_value)

        if self.chain_available:
            try:
                result = self.remote.record(hash_value, source_url)
                entry["tx_hash"] = result["tx_hash"]
                entry["block_number"] = result["block_number"]
                entry["network"] = result["network"]
                entry["status"] = result["status"]
                model = self.db.session.get(self.local.Entry, prediction_id)
                if model is not None:
                    model.tx_hash = result["tx_hash"]
                    model.block_number = result["block_number"]
                    model.network = result["network"]
                    model.status = result["status"]
            except Exception as exc:  # never break the request because of the chain
                entry["status"] = "local-only"
                entry["error"] = str(exc)

        return entry

    def status(self) -> dict:
        return {
            "local": {"available": True, "network": self.local.network},
            "blockchain": (
                self.remote.status()
                if self.remote
                else {"available": False, "reason": "not configured"}
            ),
        }


def get_service(db, entry_model) -> TraceabilityService:
    """Build the service from environment configuration."""
    rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "")
    contract = os.getenv("BLOCKCHAIN_CONTRACT_ADDRESS", "")
    remote: Optional[EthereumLedger] = None
    if rpc_url:
        try:
            remote = EthereumLedger(rpc_url, contract, os.getenv("BLOCKCHAIN_PRIVATE_KEY"))
        except Exception:
            remote = None
    return TraceabilityService(db, entry_model, remote)
