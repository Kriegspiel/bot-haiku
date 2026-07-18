"""Shared, process-safe monthly provider budget accounting."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


MICRO_USD_PER_USD = 1_000_000
STATE_VERSION = 1
DEFAULT_RESERVATION_TTL_SECONDS = 1800.0
REQUEST_ESTIMATE_SAFETY_FACTOR = 1.10


class BudgetStateError(RuntimeError):
    """Raised when the persistent budget state cannot be trusted."""


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    amount_microusd: int


@dataclass(frozen=True)
class BudgetSnapshot:
    period_utc: str
    limit_microusd: int
    spent_microusd: int
    reserved_microusd: int

    @property
    def remaining_microusd(self) -> int:
        return max(0, self.limit_microusd - self.spent_microusd - self.reserved_microusd)

    @property
    def limit_usd(self) -> float:
        return self.limit_microusd / MICRO_USD_PER_USD

    @property
    def spent_usd(self) -> float:
        return self.spent_microusd / MICRO_USD_PER_USD

    @property
    def reserved_usd(self) -> float:
        return self.reserved_microusd / MICRO_USD_PER_USD

    @property
    def remaining_usd(self) -> float:
        return self.remaining_microusd / MICRO_USD_PER_USD


class MonthlyBudgetLedger:
    """Persist a UTC-month spend cap across processes on one host."""

    def __init__(
        self,
        path: str | Path,
        *,
        limit_usd: float,
        reservation_ttl_seconds: float = DEFAULT_RESERVATION_TTL_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.limit_microusd = usd_to_microusd(limit_usd)
        self.reservation_ttl_seconds = max(60.0, float(reservation_ttl_seconds))
        self.now = now

    def status(self) -> BudgetSnapshot:
        with self._locked_state() as state:
            return self._snapshot(state)

    def reserve(self, maximum_cost_usd: float) -> BudgetReservation | None:
        amount_microusd = usd_to_microusd(maximum_cost_usd)
        if amount_microusd <= 0:
            raise ValueError("budget reservation must be greater than zero")

        with self._locked_state() as state:
            snapshot = self._snapshot(state)
            if amount_microusd > snapshot.remaining_microusd:
                return None

            reservation = BudgetReservation(uuid.uuid4().hex, amount_microusd)
            reservations = state["reservations"]
            reservations[reservation.reservation_id] = {
                "amount_microusd": amount_microusd,
                "expires_at_unix": self.now() + self.reservation_ttl_seconds,
            }
            return reservation

    def settle(self, reservation: BudgetReservation, actual_cost_usd: float | None) -> BudgetSnapshot:
        actual_microusd = reservation.amount_microusd if actual_cost_usd is None else usd_to_microusd(actual_cost_usd)
        with self._locked_state() as state:
            pending = state["reservations"].pop(reservation.reservation_id, None)
            if pending is None:
                return self._snapshot(state)
            state["spent_microusd"] += actual_microusd
            return self._snapshot(state)

    def _locked_state(self):
        return _LockedBudgetState(self)

    def _normalize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        now = self.now()
        period_utc = utc_month(now)
        if state.get("period_utc") != period_utc:
            state = new_state(period_utc)

        state["limit_microusd"] = self.limit_microusd
        reservations = state["reservations"]
        expired = [
            reservation_id
            for reservation_id, reservation in reservations.items()
            if float(reservation["expires_at_unix"]) <= now
        ]
        for reservation_id in expired:
            reservation = reservations.pop(reservation_id)
            state["spent_microusd"] += int(reservation["amount_microusd"])
        return state

    def _snapshot(self, state: dict[str, Any]) -> BudgetSnapshot:
        reserved = sum(int(item["amount_microusd"]) for item in state["reservations"].values())
        return BudgetSnapshot(
            period_utc=str(state["period_utc"]),
            limit_microusd=self.limit_microusd,
            spent_microusd=int(state["spent_microusd"]),
            reserved_microusd=reserved,
        )


class _LockedBudgetState:
    def __init__(self, ledger: MonthlyBudgetLedger) -> None:
        self.ledger = ledger
        self.lock_file = None
        self.state: dict[str, Any] | None = None

    def __enter__(self) -> dict[str, Any]:
        parent = self.ledger.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock_file = self.ledger.lock_path.open("a+", encoding="utf-8")
        os.chmod(self.ledger.lock_path, 0o600)
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        try:
            self.state = self.ledger._normalize_state(read_state(self.ledger.path, now=self.ledger.now))
        except Exception:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            raise
        return self.state

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        try:
            if exc_type is None and self.state is not None:
                write_state(self.ledger.path, self.state)
        finally:
            if self.lock_file is not None:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()


def utc_month(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m")


def usd_to_microusd(value: float) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError("USD value must be a finite non-negative number")
    return int(math.ceil(numeric * MICRO_USD_PER_USD))


def new_state(period_utc: str) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "period_utc": period_utc,
        "limit_microusd": 0,
        "spent_microusd": 0,
        "reservations": {},
    }


def read_state(path: Path, *, now: Callable[[], float]) -> dict[str, Any]:
    if not path.exists():
        return new_state(utc_month(now()))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BudgetStateError(f"cannot read provider budget state: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("version") != STATE_VERSION:
        raise BudgetStateError("unsupported provider budget state")
    if not isinstance(payload.get("period_utc"), str):
        raise BudgetStateError("provider budget period is invalid")
    if not isinstance(payload.get("spent_microusd"), int) or payload["spent_microusd"] < 0:
        raise BudgetStateError("provider budget spend is invalid")
    reservations = payload.get("reservations")
    if not isinstance(reservations, dict):
        raise BudgetStateError("provider budget reservations are invalid")
    for reservation in reservations.values():
        if not isinstance(reservation, dict):
            raise BudgetStateError("provider budget reservation is invalid")
        if not isinstance(reservation.get("amount_microusd"), int) or reservation["amount_microusd"] <= 0:
            raise BudgetStateError("provider budget reservation amount is invalid")
        expires_at = reservation.get("expires_at_unix")
        if not isinstance(expires_at, (int, float)) or not math.isfinite(float(expires_at)):
            raise BudgetStateError("provider budget reservation expiry is invalid")
    return payload


def write_state(path: Path, state: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def estimate_request_cost_upper_bound_usd(
    payload: dict[str, Any],
    *,
    input_usd_per_million_tokens: float,
    output_usd_per_million_tokens: float,
    maximum_output_tokens: int,
) -> float:
    serialized_bytes = len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
    estimated = (
        serialized_bytes * max(0.0, float(input_usd_per_million_tokens))
        + max(0, int(maximum_output_tokens)) * max(0.0, float(output_usd_per_million_tokens))
    ) / MICRO_USD_PER_USD
    return max(1 / MICRO_USD_PER_USD, estimated * REQUEST_ESTIMATE_SAFETY_FACTOR)
