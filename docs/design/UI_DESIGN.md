# SIREN — UI Design Layout

**Companion to:** `docs/spec/PRD.md` §12, `docs/spec/API_CONTRACT.md`, `docs/spec/DEVIN_BRIEFS.md` (D7).
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
┌────────────────────────────────────────────────────────────────────────┐
│ HEADER: [Review] [run-0004]  [Simple (Triage) | Advanced (Analyst)]    │
│         [ESCALATION POLICY] Advisory auto-routed to First Responders.  │
│         Public broadcast held for Human Gate confirmation.             │
├──────────────────────────────┬──────────────────────────────┬─────────┤
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

- **Simple / Advanced toggle:** Simple (Triage) mode shows a satellite-first triage card with early warning banner, heatmap, contamination event details, chlorine logistics formula, and SOS protocol checklist. Advanced (Analyst) mode shows the full evidence panel, risk gauges, and reasons list.
- **Early warning banner:** "★ Early warning 12 days — trend flagged at obs-01 before critical threshold at obs-03" (surfaces the lead time from the satellite timeline).
- **Escalation policy badge:** Shows when review is pending (elevated/critical, no decision). Communicates the two-tier routing concept: first responders get an advisory automatically, public broadcast requires human confirmation.
- **Evidence panel (left, ~40%):** before/after rasters (object-cover, full-bleed) + change overlay + swipe.
- **Risk gauge (center):** H / E / D_risk / confidence as gauge bars. Each score shows its value + a mini reason.
- **Reasons panel:** strictly ≥3 deterministic reasons (PRD §9.5). Never a bare number. Text uses `text-text-primary` for projector legibility.
- **Right dock (320px):** Disease Prevention Action Sheet (Track 7.iii) + ranked exposed-infrastructure table.
- **Decision bar (sticky bottom):** Confirm SOS (green, primary) / Reject (red) / Postpone (amber). Requires a confirmation state before Confirm fires. Reviewer identity shown.
- **Auto-SOS on CONFIRM:** Clicking CONFIRM fires a real ntfy.sh push notification to the coordinator's phone automatically (when online). Toast confirms "Decision confirmed — SOS sent to phone".

---

## 6. AuditView — Lineage & Resilient Alerting

```
┌────────────────────────────────────────────────────────────────────────┐
│ HEADER: [Audit] [AIR-GAP VERIFIED]  3 entries  [Export ▾]              │
├────────────────────────────────────────────────────────────────────────┤
│  PAYLOAD BOX (Track 7.ii)                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ {"aid":"siren-04","sec":"B","haz":"GLOF_FL",...}                 │  │
│  │  [118 / 250 bytes]  ✓ LoRa-compatible                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────┤
│  CHANNEL SIMULATOR                                                     │
│  [SMS] [LoRa Mesh] [Satellite]  →  delivery status badges             │
│  SMS: sent ✓ (live ntfy.sh) · LoRa: delivered ✓ · Satellite: queued ⏳│
│  RF telemetry: LoRa 868.1 MHz ISM · Iridium 1621 MHz L-Band           │
├────────────────────────────────────────────────────────────────────────┤
│  AUDIT TRAIL (immutable lineage table)                                │
│  [ADVISORY] system → Hospitals, Firefighters, SAR (pre-confirmation)  │
│  Timestamp (UTC) │ Actor │ Action │ Detail (JSON) │ Hash              │
│  2026-08-04T12:05 │ pipeline │ run │ {...} │ de0190c1…                │
│  2026-08-04T12:10 │ coordinator-01 │ review │ {...} │ 6deffb9b…       │
│  2026-08-04T12:11 │ coordinator-01 │ dispatch │ {...} │ 382e377c…     │
│  [VERIFY CHAIN] → Web Crypto SHA-256 verification modal               │
└────────────────────────────────────────────────────────────────────────┘
```

- **AIR-GAP VERIFIED badge:** Green badge in header indicating offline-first operation.
- **Export dropdown:** Ledger JSON export + SitRep TXT export (field situation report).
- **Payload box:** raw compressed JSON + byte counter (must show ≤250). Green badge when compliant.
- **Channel simulator:** SMS / LoRa Mesh / Satellite with live status badges. SMS fires a real ntfy.sh push when online (gated by `navigator.onLine`). LoRa and Satellite are simulated state machines (QUEUED → TRANSMITTING → DELIVERED).
- **RF telemetry specs:** LoRa (868.1 MHz ISM, SF9, 125 kHz, 222 bytes max) and Iridium SBD (1621 MHz L-Band, 340 bytes/SBD).
- **First Responder Advisory row:** When severity is elevated/critical and no confirmation yet, a pre-confirmation advisory row appears at the top of the audit trail. Amber border, labeled "ADVISORY" not "DISPATCH". Hash column shows "simulated". Disappears once confirmation is recorded.
- **Audit trail:** append-only table — timestamp, actor, action, JSON detail, hash. Monospace for JSON. Real SHA-256 hashes (not placeholders).
- **Verify Chain:** Web Crypto API (`crypto.subtle.digest`) verification modal. Recomputes SHA-256 for each entry and checks the chain. Shows "ALL 3 BLOCKS CRYPTOGRAPHICALLY LINKED" + "0 TAMPERING DETECTED" on success.
- **Secondary SEND TO PHONE:** Manual ntfy.sh send button (in addition to the auto-fire on CONFIRM in ReviewView).

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

---

## 10. Implemented Additions (post-initial-design)

The following were added after the initial UI design spec was written:

- **Multi-theme system** (`frontend/src/theme/`): `ThemeContext.tsx` + `ThemeToggle.tsx` providing Ops Dark (default), Professional Light, and Satellite themes. The toggle is a compact chip in the header bar.
- **OfflineBadge** (`frontend/src/components/OfflineBadge.tsx`): Truthful online/offline indicator using `navigator.onLine` + window `online`/`offline` event listeners. Shows "ONLINE" or "OFFLINE — ALL SYSTEMS LOCAL".
- **Projector-ready typography scaling**: Centralized type tokens in `tailwind.config.js` + `index.css` for projector legibility. Nav height 54px, banner 46px, label-caps 14px/600 weight.
- **ReviewView Simple/Advanced toggle**: Simple (Triage) mode shows satellite-first triage card with early warning banner, heatmap, chlorine logistics, and SOS checklist. Advanced (Analyst) mode shows full evidence panel, gauges, and reasons.
- **Auto-SOS on CONFIRM**: Clicking CONFIRM in ReviewView fires a real ntfy.sh push notification automatically. Shared utility in `frontend/src/utils/ntfy.ts`.
- **Escalation policy badge**: Static badge in ReviewView header showing two-tier routing concept (first responders advisory vs. public broadcast held for human gate).
- **Early warning banner**: "★ Early warning 12 days" on SimpleTriage, surfacing the satellite timeline lead time.
- **AuditView enhancements**:
  - Real SHA-256 mock hashes (computed using the backend formula, not placeholders)
  - Ledger JSON export + SitRep TXT export via Export dropdown
  - Web Crypto verification modal (`crypto.subtle.digest`) — verifies the full hash chain in-browser
  - RF telemetry specs (LoRa 868.1 MHz ISM, Iridium 1621 MHz L-Band)
  - First Responder Advisory row (pre-confirmation, simulated, amber border)
  - ntfy.sh live phone alerts (gated by `navigator.onLine`)
  - Secondary SEND TO PHONE button for manual re-send
- **Audit run_id wiring**: AuditView queries `GET /audit?run_id={activeRunId}` when an active run exists, falling back to mock data only on network failure. The terminal SHA-256 digest is rendered from the last audit entry's `event_hash`.
- **Audit preview modal**: Decoded plain-text emergency handset alert format, opened via a [PREVIEW] button. Escape closes the modal first (consumes `siren:escape` event).
- **Channel FSM**: AuditView dispatch channels cycle through QUEUED → TRANSMITTING → DELIVERED (or QUEUED for satellite) with timed transitions.
- **MapView polish**: Circular asset markers (border-radius:50%) + circular legend dots. Initial camera uses `jumpTo` (centered on Imja Lake) instead of `fitBounds`. Solid 3px corridor line (was thin dashed). Swipe compare uses absolute-positioned object-cover for clean before/after reveal.
- **TimelineView polish**: Single legend in chart header (removed duplicate SVG labels). Axis font sizes bumped (Y 11px, X 12px). Thumbnails use `object-cover` (no letterboxing voids).
- **Tailwind CSS**: The design system is implemented with Tailwind utility classes + CSS custom properties. No `styles/tokens.css` file — tokens live in `index.css` and Tailwind config.