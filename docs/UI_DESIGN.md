# SIREN — UI Design Layout

**Companion to:** `docs/PRD.md` §12, `docs/API_CONTRACT.md`, `docs/DEVIN_BRIEFS.md` (D7).
**Purpose:** Defines the visual layout, component hierarchy, and design system for the coordinator console. The frontend scaffold (D7) builds against this.

---

## 1. Design Principles

1. **Dark, high-contrast ops console.** Emergency coordinators monitor for long shifts; dark surfaces reduce eye strain and make alert colors pop. This is a *command console*, not a marketing site.
2. **Status-color driven.** Green / amber / red carry meaning everywhere (assets, alerts, dispatch). Never use them decoratively.
3. **The map is the hero.** MapView is full-bleed. All other views are supporting panels.
4. **The review card is unmissable.** When severity is elevated/critical, a persistent alert banner + the ReviewView decision bar dominate the screen.
5. **Information-dense but scannable.** Coordinators read fast. Use compact tables, badges, and gauges — not prose.

---

## 2. App Shell (all views share this)

```
┌────────────────────────────────────────────────────────────────────────┐
│ NAV BAR: [SIREN ▾]  Basin: Dudh Koshi/Imja  ● LIVE  | Map Timeline Review Audit │
├────────────────────────────────────────────────────────────────────────┤
│ ALERT BANNER (only when severity = elevated/critical)                  │
│   ⚠ ELEVATED — Imja lake expansion +28% · 2 villages · 1 bridge · 3 wells │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│                        ACTIVE VIEW CONTENT                             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

- **Nav bar (48px):** brand mark + basin selector dropdown, live-status dot, 4 view tabs.
- **Alert banner (auto-hide, 40px):** slides in when a run produces `severity ∈ {elevated, critical}`. Clicking it jumps to ReviewView. This is the "can't miss it" element.
- **View tabs:** Map (default) · Timeline · Review · Audit.

---

## 3. MapView — The Core Visual Canvas

```
┌──────────────────────────────────────────────┬─────────────────────────┐
│  MAP (full-bleed MapLibre canvas)            │  RIGHT DOCK (280px)     │
│  · basin polygon overlay                     │  ┌─────────────────────┐│
│  · D8 flowlines                              │  │ ASSET LEGEND        ││
│  · water expansion mask                      │  │  ● green = safe     ││
│  · OSM asset markers                         │  │  ● amber = buffered ││
│  · before/after swipe                        │  │  ● red = inundated  ││
│                                              │  └─────────────────────┘│
│  ┌──────────────┐  ┌──────────────────────┐  │  ┌─────────────────────┐│
│  │ LEFT DOCK     │  │ TOP OVERLAY          │  │  │ SELECTED ASSET      ││
│  │ layer toggles │  │ basin + obs badge    │  │  │ detail card         ││
│  └──────────────┘  └──────────────────────┘  │  └─────────────────────┘│
│                                              │                         │
│  ┌──────────────────────────────────────────┐│                         │
│  │ BOTTOM: swipe-compare / opacity slider   ││                         │
│  └──────────────────────────────────────────┘│                         │
└──────────────────────────────────────────────┴─────────────────────────┘
```

- **Map center:** [86.82, 27.88], zoom 11.
- **Left dock (collapsible, 220px):** layer toggles — Basin AOI, DEM hillshade, Optical baseline, SAR backscatter, D8 flowlines, Water expansion, OSM assets.
- **Right dock (280px):** asset legend + selected-asset detail card (name, type, population, status, distance). Clicking a marker populates this card.
- **Bottom control:** before/after swipe (drag handle) OR opacity slider — the demo's dramatic beat.
- **Asset markers:** green (safe) / amber (within 100m hazard buffer) / red (inundated).

---

## 4. TimelineView — Observation Sequence & Run Controller

```
┌────────────────────────────────────────────────────────────────────────┐
│  SIMULATION CONTROLLER (top strip)                                     │
│  [▶ Run Simulation]  ·  status: before → disaster-day  ·  progress bar │
├────────────────────────────────────────────────────────────────────────┤
│  WEATHER-ADAPTIVE ROUTER STRIP                                         │
│  Obs 2: Optical cloud 95% → SWITCHED TO SAR PATH  [badge]             │
├────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                      │
│  │ Baseline│ │ Obs 1   │ │ Obs 2   │ │ Obs 3   │   ← horizontal cards │
│  │ 2025-11 │ │ 2026-07 │ │ 2026-08 │ │ 2026-08 │                      │
│  │ S2 opt  │ │ S1 SAR  │ │ S1 SAR  │ │ S1 SAR  │                      │
│  │ cloud 5%│ │ cloud 0%│ │cloud 95%│ │cloud 90%│                      │
│  │ rain 0  │ │ rain 18 │ │ rain 85 │ │ rain 60 │                      │
│  │ area 3.0│ │ area 3.2│ │ area 4.1│ │ area 4.3│                      │
│  │ +0%     │ │ +8%     │ │ +28%    │ │ +43%    │                      │
│  │ SAFE    │ │ ADVISORY│ │ELEVATED │ │CRITICAL │                      │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                      │
└────────────────────────────────────────────────────────────────────────┘
```

- **Simulation controller:** the demo's primary control. Starts in the **before** state (baseline only, all assets green). Clicking **Run Simulation** advances through the observations to the disaster day, revealing the +8% → +28% expansion and the escalating severity.
- **Prevention callout:** after the run, a banner highlights "12 days of warning between Obs 1 (+8%) and Obs 2 (+28%)" — the retrospective prevention story.
- **Router strip:** shows the optical→SAR switch (the demo's technical proof).
- **Observation cards:** compact, each showing date, sensor badge, cloud %, 24h/7d rain, water area, % change, severity chip. Severity chip color-coded (safe=green, advisory=amber, elevated=orange, critical=red).
- **Trend arrow:** ▲/▼/→ per card.

---

## 5. ReviewView — Human-in-the-Loop Coordinator Console

```
┌──────────────────────────────┬──────────────────────────────┬─────────┐
│  EVIDENCE PANEL (left)       │  RISK GAUGE (center)         │ RIGHT   │
│  · before/after rasters      │  ┌────────────────────────┐  │ (320px) │
│  · change overlay            │  │  H 0.62  E 0.48        │  │┌───────┐│
│  · swipe compare             │  │  D 0.31  conf 0.76     │  ││Disease││
│                              │  │  [gauge bars]          │  ││Action ││
│                              │  └────────────────────────┘  ││Sheet  ││
│                              │  REASONS (≥3, deterministic) ││       ││
│                              │  · reason 1                  ││┌──────┐││
│                              │  · reason 2                  │││Assets│││
│                              │  · reason 3                  │││table │││
│                              │                              ││└──────┘││
│                              │                              │└───────┘│
├──────────────────────────────┴──────────────────────────────┴─────────┤
│  DECISION BAR (sticky bottom, always visible)                          │
│  [✓ Confirm SOS]  [✗ Reject]  [⏸ Postpone]   ·  reviewer: coordinator-01│
└────────────────────────────────────────────────────────────────────────┘
```

- **Evidence panel (left, ~40%):** before/after rasters + change overlay + swipe.
- **Risk gauge (center):** H / E / D_risk / confidence as gauge bars. Each score shows its value + a mini reason.
- **Reasons panel:** strictly ≥3 deterministic reasons (PRD §9.5). Never a bare number.
- **Right dock (320px):** Disease Prevention Action Sheet (Track 7.iii) + ranked exposed-infrastructure table.
- **Decision bar (sticky bottom):** Confirm SOS (green, primary) / Reject (red) / Postpone (amber). Requires a confirmation state before Confirm fires. Reviewer identity shown.

---

## 6. AuditView — Lineage & Resilient Alerting

```
┌────────────────────────────────────────────────────────────────────────┐
│  PAYLOAD BOX (Track 7.ii)                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ {"aid":"siren-04","sec":"B","haz":"GLOF_FL",...}                 │  │
│  │  [118 / 250 bytes]  ✓ LoRa-compatible                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────┤
│  CHANNEL SIMULATOR                                                     │
│  [SMS] [LoRa Mesh] [Satellite]  →  delivery status badges             │
│  SMS: sent ✓ · LoRa: delivered ✓ · Satellite: queued ⏳               │
├────────────────────────────────────────────────────────────────────────┤
│  AUDIT TRAIL (immutable lineage table)                                │
│  Timestamp (UTC) │ Actor │ Action │ Detail (JSON)                     │
│  2026-08-04T12:05 │ pipeline │ run │ {...}                            │
│  2026-08-04T12:10 │ coordinator-01 │ review │ {...}                   │
│  2026-08-04T12:11 │ coordinator-01 │ dispatch │ {...}                 │
└────────────────────────────────────────────────────────────────────────┘
```

- **Payload box:** raw compressed JSON + byte counter (must show ≤250). Green badge when compliant.
- **Channel simulator:** SMS / LoRa Mesh / Satellite with live status badges.
- **Audit trail:** append-only table — timestamp, actor, action, JSON detail. Monospace for the JSON.

---

## 7. Design System

### Color palette (dark theme)
| Token | Hex | Usage |
|---|---|---|
| `bg` | `#0F172A` | App background (slate-900) |
| `panel` | `#1E293B` | Panels/cards (slate-800) |
| `panel-2` | `#334155` | Nested panels, borders (slate-700) |
| `text` | `#E2E8F0` | Primary text (slate-200) |
| `text-dim` | `#94A3B8` | Secondary text (slate-400) |
| `accent` | `#06B6D4` | SIREN brand, active states (cyan-500) |
| `safe` | `#22C55E` | Green — safe assets, success |
| `warn` | `#F59E0B` | Amber — buffered risk, advisory |
| `danger` | `#EF4444` | Red — inundated, critical, reject |
| `info` | `#3B82F6` | Blue — informational |

### Typography
- **UI:** system stack (`Inter`, `-apple-system`, `Segoe UI`, sans-serif).
- **Telemetry/payload/JSON:** monospace (`JetBrains Mono`, `SFMono-Regular`, monospace).
- **Sizes:** 12px (labels/badges), 14px (body), 16px (card titles), 20px (view titles), 28px (gauge values).

### Spacing & radius
- 8px grid. Panel padding 16px. Card gap 12px.
- Border radius 8px (panels), 4px (badges/buttons).

### Status chips
- `advisory` → amber outline · `elevated` → orange fill · `critical` → red fill · `informational` → blue outline.

---

## 8. Component Hierarchy

```text
App
├── NavBar (brand, basin selector, live dot, view tabs)
├── AlertBanner (severity-driven, click → ReviewView)
└── ViewRouter
    ├── MapView
    │   ├── MapCanvas (MapLibre)
    │   ├── LayerTogglePanel (left dock)
    │   ├── AssetLegend (right dock)
    │   ├── AssetDetailCard (right dock)
    │   └── SwipeCompare (bottom)
    ├── TimelineView
    │   ├── RunController (Run Monitoring button)
    │   ├── RouterStrip (weather-adaptive badge)
    │   └── ObservationCard[] (horizontal)
    ├── ReviewView
    │   ├── EvidencePanel (before/after + overlay)
    │   ├── RiskGauge (H/E/D/confidence)
    │   ├── ReasonsPanel (≥3)
    │   ├── DiseaseActionSheet (Track 7.iii)
    │   ├── ExposedAssetsTable
    │   └── DecisionBar (Confirm/Reject/Postpone)
    └── AuditView
        ├── PayloadBox (byte counter)
        ├── ChannelSimulator (SMS/LoRa/Satellite)
        └── AuditTrailTable (immutable lineage)
```

---

## 9. Demo-Critical Visual Beats (must render correctly)

1. **The swipe beat:** MapView before/after swipe reveals the +28% SAR water surge cutting through clouds.
2. **The router badge:** TimelineView shows "Optical cloud 95% → Switched to SAR Path."
3. **The action sheet:** ReviewView lists 3 submerged wells + chlorine quotas + boil-water notice.
4. **The payload box:** AuditView shows `118 / 250 bytes` with a green LoRa-compatible badge.
5. **The decision bar:** Confirm SOS is unmissable and requires confirmation state.