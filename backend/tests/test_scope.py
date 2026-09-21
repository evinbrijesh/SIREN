"""Tests for the operational scope declaration (vulnerable-season window)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from siren.scope import (
    EVENT_MONTH_DISTRIBUTION,
    EVENT_TOTAL,
    OUT_OF_SCOPE_NOTE,
    VULNERABLE_MONTHS,
    in_operational_window,
    scope_for_date,
    scope_summary,
)


class TestWindow:
    def test_window_is_june_to_september(self):
        assert VULNERABLE_MONTHS == (6, 7, 8, 9)

    @pytest.mark.parametrize("month", [6, 7, 8, 9])
    def test_months_in_window(self, month):
        assert in_operational_window(date(2026, month, 15))

    @pytest.mark.parametrize("month", [1, 2, 3, 4, 5, 10, 11, 12])
    def test_months_outside_window(self, month):
        assert not in_operational_window(date(2026, month, 15))

    def test_accepts_iso_string(self):
        assert in_operational_window("2025-08-29")
        assert not in_operational_window("2026-01-08")

    def test_accepts_datetime(self):
        assert in_operational_window(datetime(2025, 9, 10, 12, 30))
        assert not in_operational_window(datetime(2025, 11, 21, 6, 0))


class TestScopeForDate:
    def test_in_scope_payload(self):
        scope = scope_for_date("2025-09-10")
        assert scope["in_scope"] is True
        assert scope["month"] == 9
        assert "monsoon" in scope["window"].lower()
        assert scope["event_share_in_window"] == pytest.approx(0.826, abs=0.001)

    def test_out_of_scope_payload(self):
        scope = scope_for_date("2026-01-20")
        assert scope["in_scope"] is False
        assert scope["reason"] == OUT_OF_SCOPE_NOTE
        assert "deterministic baseline remains authoritative" in scope["reason"]

    def test_reason_strings_differ(self):
        assert scope_for_date("2025-07-01")["reason"] != scope_for_date(
            "2025-12-01"
        )["reason"]


class TestDerivationEvidence:
    def test_event_distribution_sums_to_total(self):
        assert sum(EVENT_MONTH_DISTRIBUTION.values()) == EVENT_TOTAL

    def test_majority_of_events_in_window(self):
        in_window = sum(EVENT_MONTH_DISTRIBUTION[m] for m in VULNERABLE_MONTHS)
        assert in_window / EVENT_TOTAL > 0.80

    def test_scope_summary_shape(self):
        summary = scope_summary()
        assert summary["months"] == list(VULNERABLE_MONTHS)
        assert summary["derivation"]["gloF_events"]["total"] == EVENT_TOTAL
        assert summary["derivation"]["rainfall"]["mm_per_month"][7] == 144
        assert "orbit" in summary["other_axes"]
