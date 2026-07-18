from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from provider_budget import BudgetStateError, MonthlyBudgetLedger, estimate_request_cost_upper_bound_usd


class MutableClock:
    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp

    def __call__(self) -> float:
        return self.timestamp


class MonthlyBudgetLedgerTests(unittest.TestCase):
    def test_reservations_are_shared_and_settle_to_actual_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openai.json"
            first = MonthlyBudgetLedger(path, limit_usd=18)
            second = MonthlyBudgetLedger(path, limit_usd=18)

            reservation = first.reserve(2.5)
            self.assertIsNotNone(reservation)
            self.assertAlmostEqual(second.status().reserved_usd, 2.5)

            snapshot = second.settle(reservation, 0.75)  # type: ignore[arg-type]
            self.assertAlmostEqual(snapshot.spent_usd, 0.75)
            self.assertAlmostEqual(snapshot.reserved_usd, 0)
            self.assertAlmostEqual(snapshot.remaining_usd, 17.25)

    def test_reservation_refuses_to_exceed_monthly_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = MonthlyBudgetLedger(Path(tmp) / "anthropic.json", limit_usd=18)
            reservation = ledger.reserve(17.5)
            self.assertIsNotNone(reservation)
            self.assertIsNone(ledger.reserve(0.500001))
            self.assertIsNotNone(ledger.reserve(0.5))

    def test_unknown_call_cost_charges_full_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = MonthlyBudgetLedger(Path(tmp) / "openai.json", limit_usd=18)
            reservation = ledger.reserve(1.25)
            self.assertIsNotNone(reservation)
            snapshot = ledger.settle(reservation, None)  # type: ignore[arg-type]
            self.assertAlmostEqual(snapshot.spent_usd, 1.25)

    def test_expired_reservations_are_charged_and_late_settlement_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = MutableClock(1_750_000_000)
            ledger = MonthlyBudgetLedger(
                Path(tmp) / "openai.json",
                limit_usd=18,
                reservation_ttl_seconds=60,
                now=clock,
            )
            reservation = ledger.reserve(4)
            self.assertIsNotNone(reservation)
            clock.timestamp += 61
            self.assertAlmostEqual(ledger.status().reserved_usd, 0)
            self.assertAlmostEqual(ledger.status().spent_usd, 4)
            self.assertAlmostEqual(ledger.status().remaining_usd, 14)
            late_snapshot = ledger.settle(reservation, 1)  # type: ignore[arg-type]
            self.assertAlmostEqual(late_snapshot.spent_usd, 4)

    def test_new_utc_month_resets_spend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = MutableClock(1_751_327_900)  # 2025-06-30 UTC
            ledger = MonthlyBudgetLedger(Path(tmp) / "openai.json", limit_usd=18, now=clock)
            reservation = ledger.reserve(2)
            self.assertIsNotNone(reservation)
            ledger.settle(reservation, 2)  # type: ignore[arg-type]

            clock.timestamp = 1_751_328_100  # 2025-07-01 UTC
            snapshot = ledger.status()
            self.assertEqual(snapshot.period_utc, "2025-07")
            self.assertAlmostEqual(snapshot.spent_usd, 0)
            self.assertAlmostEqual(snapshot.remaining_usd, 18)

    def test_corrupt_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openai.json"
            path.write_text("not-json", encoding="utf-8")
            ledger = MonthlyBudgetLedger(path, limit_usd=18)
            with self.assertRaises(BudgetStateError):
                ledger.status()

    def test_threaded_reservations_cannot_collectively_exceed_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openai.json"
            reservations = []
            lock = threading.Lock()

            def reserve() -> None:
                reservation = MonthlyBudgetLedger(path, limit_usd=18).reserve(1)
                if reservation is not None:
                    with lock:
                        reservations.append(reservation)

            threads = [threading.Thread(target=reserve) for _ in range(30)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

            self.assertEqual(len(reservations), 18)
            self.assertAlmostEqual(MonthlyBudgetLedger(path, limit_usd=18).status().reserved_usd, 18)

    def test_request_cost_estimate_uses_serialized_bytes_and_output_cap(self) -> None:
        payload = {"input": "hello", "max_output_tokens": 100}
        estimate = estimate_request_cost_upper_bound_usd(
            payload,
            input_usd_per_million_tokens=2,
            output_usd_per_million_tokens=10,
            maximum_output_tokens=100,
        )
        raw_bytes = len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
        self.assertGreaterEqual(estimate, (raw_bytes * 2 + 100 * 10) / 1_000_000)


if __name__ == "__main__":
    unittest.main()
