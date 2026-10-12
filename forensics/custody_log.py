"""Append-only, hash-chained chain-of-custody log.

Each line is one JSON entry carrying the SHA-256 of the previous entry
(``prev_hash``) and of its own content (``entry_hash``). Editing, deleting or
reordering any past line breaks the chain, which :meth:`CustodyLog.verify`
detects. This makes tampering *evident*, not impossible -- whoever controls
the file can rewrite the whole chain -- so for real casework the log file
should itself be backed up or countersigned off-machine.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GENESIS_HASH = "0" * 64

_append_lock = threading.Lock()


def _entry_digest(entry_without_hash: dict) -> str:
    """Hash an entry's canonical JSON form (sorted keys, no whitespace)."""
    canonical = json.dumps(entry_without_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying a custody log.

    Attributes:
        is_valid: ``True`` if every entry's hash and chain link check out.
        entry_count: Number of entries read.
        first_bad_line: 1-based line number of the first broken entry, or ``None``.
        reason: Why verification failed, or ``""``.
    """

    is_valid: bool
    entry_count: int
    first_bad_line: int | None = None
    reason: str = ""


class CustodyLog:
    """Append and verify entries in one JSON-lines custody log file."""

    def __init__(self, log_path: Path) -> None:
        """Bind to a log file (created on first append).

        Args:
            log_path: Path of the ``.jsonl`` custody log.
        """
        self.log_path = Path(log_path)

    def _last_hash(self) -> str:
        """Return the ``entry_hash`` of the last entry, or the genesis hash for an empty log."""
        if not self.log_path.exists():
            return GENESIS_HASH
        last_line = ""
        with self.log_path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                if line.strip():
                    last_line = line
        return json.loads(last_line)["entry_hash"] if last_line else GENESIS_HASH

    def append(self, action: str, details: dict) -> dict:
        """Append one entry, chained to the previous one.

        Args:
            action: What happened, e.g. ``"analysis"``.
            details: JSON-serializable fields describing the event (case ID,
                examiner, evidence hash, verdict, ...).

        Returns:
            The full entry as written, including ``prev_hash`` and ``entry_hash``.
        """
        with _append_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "action": action,
                **details,
                "prev_hash": self._last_hash(),
            }
            entry["entry_hash"] = _entry_digest(entry)
            with self.log_path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def verify(self) -> VerificationResult:
        """Recompute every entry hash and chain link.

        Returns:
            A :class:`VerificationResult`. An absent log is valid with zero entries.
        """
        if not self.log_path.exists():
            return VerificationResult(is_valid=True, entry_count=0)

        expected_prev = GENESIS_HASH
        entry_count = 0
        with self.log_path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                if not line.strip():
                    continue
                entry_count += 1
                try:
                    entry = json.loads(line)
                    recorded_hash = entry.pop("entry_hash")
                except (json.JSONDecodeError, KeyError):
                    return VerificationResult(False, entry_count, line_number, "Entry is not valid JSON or has no hash.")
                if entry.get("prev_hash") != expected_prev:
                    return VerificationResult(
                        False, entry_count, line_number, "Chain broken: entry does not link to the previous one."
                    )
                if _entry_digest(entry) != recorded_hash:
                    return VerificationResult(
                        False, entry_count, line_number, "Entry content was modified after it was written."
                    )
                expected_prev = recorded_hash
        return VerificationResult(is_valid=True, entry_count=entry_count)
