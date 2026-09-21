"""Operational scope — the vulnerable-season window.

SIREN's neural water/change evidence is certified for the **monsoon
window only**. This module is the machine-readable declaration of that
scope: which months are operationally in scope, why, and how a given
observation is classified.

Derivation (2026-09-21), grounded in two independent datasets:

1. **Real GLOF events** — ``data/datasets/dynamic_escalation_train.parquet``
   (230 dated Himalayan breach events, HMAGLOFDB-derived):

       Jun  40 (17.4%)   Jul  77 (33.5%)   Aug  62 (27.0%)   Sep  11 (4.8%)
       Jun–Sep total: 190/230 = 82.6%      Jul–Aug: 60.5%

2. **Rainfall climatology** — ``data/assets/imja_power_series.json``
   (NASA POWER daily, 1981–2026, Imja basin):

       Jun  72 mm/mo   Jul 144 mm/mo   Aug 133 mm/mo   Sep  70 mm/mo
       Jun–Sep total: ~419 mm ≈ 80% of annual precipitation
       Jun–Sep mean temperature: +3.1 to +5.6 °C (above freezing —
       the lake surface is liquid, i.e. the model's trained regime)

Both datasets independently select **June–September (JJAS)**. Outside
this window the lake is typically frozen or refreezing (the C-band SAR
cannot see liquid water through ice — physics, not model failure), the
rainfall trigger is absent, and <18% of historical events occur.

Out-of-scope observations are not "unsupported" — the deterministic
baseline still runs and remains authoritative; the neural evidence is
annotated as outside the certified window and the reason is surfaced.

The window is a *prioritisation*, not a hard off-switch: a GLOF can
occur outside it (the 2023 South Lhonak burst was 2023-10-03, outside
JJAS — landslide-triggered). Operations should treat out-of-window
observations as lower-priority monitoring, not as impossible.

Usage:
    from siren.scope import in_operational_window, scope_for_date

    scope_for_date("2026-08-19")
    # {"in_scope": True, "window": "Jun-Sep", "reason": ...}
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

#: Months in the certified operational window (June–September, JJAS).
VULNERABLE_MONTHS: tuple[int, ...] = (6, 7, 8, 9)

WINDOW_LABEL = "Jun-Sep (monsoon)"

#: Evidence for the window derivation — kept in-code so the declaration
#: travels with the artefact it justifies.
EVENT_MONTH_DISTRIBUTION: dict[int, int] = {
    1: 1, 2: 0, 3: 3, 4: 7, 5: 16, 6: 40, 7: 77, 8: 62, 9: 11, 10: 8, 11: 2, 12: 3
}
EVENT_TOTAL = 230
EVENT_SHARE_IN_WINDOW = 190 / EVENT_TOTAL  # 0.826

#: Monthly rainfall climatology (mm/month, NASA POWER 1981–2026, Imja).
RAINFALL_MM_PER_MONTH: dict[int, float] = {
    1: 4, 2: 6, 3: 13, 4: 18, 5: 37, 6: 72, 7: 144, 8: 133, 9: 70, 10: 19, 11: 2, 12: 3
}

OUT_OF_SCOPE_NOTE = (
    "Outside the certified monsoon window (Jun-Sep): the neural "
    "water/change evidence is advisory-only and the deterministic "
    "baseline remains authoritative. The lake is typically frozen or "
    "refreezing and <18% of historical GLOF events occur here."
)

IN_SCOPE_NOTE = (
    "Within the certified monsoon window (Jun-Sep): the neural "
    "water/change evidence applies under its promoted scope "
    "(descending orbit, liquid surface)."
)


def _as_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)[:10]).date()


def in_operational_window(d: date | datetime | str) -> bool:
    """True when ``d`` falls in the certified monsoon window (Jun–Sep)."""
    return _as_date(d).month in VULNERABLE_MONTHS


def scope_for_date(d: date | datetime | str) -> dict[str, Any]:
    """Machine-readable scope classification for an observation date.

    Returns a dict with ``in_scope``, the window label, the month, and
    the reason — suitable for persisting to ``change_stats`` and
    surfacing on the review card.
    """
    day = _as_date(d)
    in_scope = day.month in VULNERABLE_MONTHS
    return {
        "in_scope": in_scope,
        "window": WINDOW_LABEL,
        "months": list(VULNERABLE_MONTHS),
        "month": day.month,
        "reason": IN_SCOPE_NOTE if in_scope else OUT_OF_SCOPE_NOTE,
        "event_share_in_window": round(EVENT_SHARE_IN_WINDOW, 3),
    }


def scope_summary() -> dict[str, Any]:
    """The full scope declaration — for docs, APIs, and audit."""
    return {
        "window": WINDOW_LABEL,
        "months": list(VULNERABLE_MONTHS),
        "derivation": {
            "gloF_events": {
                "source": "data/datasets/dynamic_escalation_train.parquet",
                "total": EVENT_TOTAL,
                "in_window": int(round(EVENT_SHARE_IN_WINDOW * EVENT_TOTAL)),
                "share": round(EVENT_SHARE_IN_WINDOW, 3),
            },
            "rainfall": {
                "source": "data/assets/imja_power_series.json",
                "mm_per_month": RAINFALL_MM_PER_MONTH,
                "window_share_of_annual": 0.80,
            },
            "surface_state": "liquid (Jun-Sep mean +3 to +6 °C at lake altitude)",
        },
        "other_axes": {
            "orbit": "descending (promoted component scope)",
            "region": "Imja-area monitorable lakes (certified checkpoint scope)",
        },
        "out_of_scope_policy": OUT_OF_SCOPE_NOTE,
    }
