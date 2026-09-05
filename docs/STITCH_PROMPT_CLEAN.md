# Stitch Design Prompt — SIREN Coordinator Console (Clean Version)

## Project Overview

Design a **dark-themed emergency operations console** called SIREN (Satellite-Informed Risk & Emergency Network). It is a disaster-response decision platform for Himalayan glacial basins. An emergency coordinator uses it to monitor satellite observations of a glacial lake, review escalating hazard evidence, confirm alerts, and dispatch compressed warnings to downstream communities.

The app has **4 views** (Map, Timeline, Review, Audit) plus a **shared app shell** with a navigation bar and alert banner. It is a **single-page application** — all views share the same shell, switched via tabs.

**Design philosophy:** Clarity over spectacle. This is a tool someone uses during a disaster. Every element on screen must serve the decision chain: *satellite saw change → system flagged it → human reviewed evidence → human confirmed → compressed alert sent.* If an element doesn't serve that chain, it shouldn't be on screen.

**Vibe:** Dark, calm, professional. Think a clean monitoring dashboard — not a sci-fi movie cockpit. Generous spacing. One clear visual hierarchy per view. Status colors carry meaning; everything else is neutral.

---

## Design System

### Color Palette (dark theme)

| Token | Hex | Usage |
|---|---|---|
| Background | `#0F172A` | App background |
| Panel | `#1E293B` | Cards, panels, docks |
| Border | `#334155` | Panel borders, dividers |
| Text | `#E2E8F0` | Primary text |
| Text-dim | `#94A3B8` | Labels, secondary text |
| Accent | `#06B6D4` | SIREN brand, active states |
| Safe | `#22C55E` | Safe status, confirmed |
| Warn | `#F59E0B` | Advisory, buffered |
| Elevated | `#F97316` | Elevated severity |
| Danger | `#EF4444` | Critical, inundated, rejected |
| Info | `#3B82F6` | Informational, optical sensor |

No glow effects. No pulse animations except on the single most critical element per view (if any). No decorative gradients. Solid colors and clean borders only.

### Typography

- **UI font:** Inter
- **Monospace font:** JetBrains Mono — used only for: JSON payloads, timestamps in tables, file paths, byte counts
- **Sizes:** 12px (labels), 14px (body), 16px (card titles), 20px (view titles), 24px (gauge values)
- **No uppercase label-caps style.** Headers are sentence case. Badges are uppercase but normal weight, not tracked-out.

### Spacing & Radius

- 8px grid. Panel padding 16px. Card gap 16px.
- Border radius: 8px (panels), 4px (badges/buttons)
- Borders: 1px solid `#334155` on panels. No drop shadows. No glows.

### Status Badges

Compact, understated. 4px radius, 2px×8px padding, 12px text.
- `SAFE` → green text, green border, transparent background
- `WATCH` → amber text, amber border, transparent background
- `ELEVATED` → orange text, orange border, transparent background
- `CRITICAL` → red text, red border, transparent background
- `CONFIRMED` → green text, green border, transparent background
- `REJECTED` → red text, red border, transparent background

No filled backgrounds on badges. No pulsing. The color itself is the signal.

---

## Page 1: App Shell (shared across all views)

### Navigation Bar (56px height, full width, border-bottom 1px)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ◉ SIREN    Dudh Koshi / Imja       [Map] [Timeline] [Review] [Audit]    │
└──────────────────────────────────────────────────────────────────────────┘
```

**Left:**
- "◉ SIREN" — brand text, 16px, accent color, medium weight. The dot is a small circle, not a logo.
- "Dudh Koshi / Imja" — basin name, 14px, text-dim

**Right:**
- 4 tab buttons: Map, Timeline, Review, Audit
- Active tab: accent color text, 2px accent bottom border
- Inactive tabs: text-dim color, no border
- 14px text, 12px vertical padding

**No live status dot. No offline badge. No reset button. No keyboard hints.** Keep it clean.

### Alert Banner (44px, slides in below nav, only when severity is elevated or critical)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ⚠  ELEVATED  ·  +14.3% water expansion detected  ·  Review →           │
└──────────────────────────────────────────────────────────────────────────┘
```

- Left border: 3px solid, severity color
- "⚠" icon + severity word in severity color, 14px medium weight
- Brief description in text color, 14px
- "Review →" link in accent color, 14px
- Background: panel color (`#1E293B`), not translucent
- Clicking navigates to Review tab

### Footer (28px, bottom of screen, border-top 1px)

```
Sentinel-2 · Sentinel-1 · SRTM · Open-Meteo · © OSM · pipeline v0.1.0
```

- 11px, text-dim, centered
- That's it. No status indicators, no latency, no UTC clock.

---

## Page 2: Map View (default landing view)

A full-bleed map with two side docks. The map is the hero — the docks are minimal.

### Layout (3-column flex, 12px gap)

```
┌──────────────┬────────────────────────────────┬─────────────────┐
│ LEFT DOCK    │  MAP CANVAS                    │ RIGHT DOCK      │
│ 200px        │  (flex-1)                      │ 240px           │
│              │                                │                 │
│ Layers       │  (map area)                    │ Asset Legend    │
│              │                                │                 │
│              │  ┌──────────────────────────┐  │ Selected Asset  │
│              │  │ Before / After Swipe     │  │ (if clicked)    │
│              │  └──────────────────────────┘  │                 │
└──────────────┴────────────────────────────────┴─────────────────┘
```

### Left Dock — Layers (200px, panel styled, 16px padding)

**Title:** "Layers" (16px, text color)

**6 toggle rows** (checkbox + label, 13px, 8px vertical spacing):
1. "Basin AOI" — checked
2. "Optical baseline" — checked
3. "SAR backscatter" — unchecked, disabled with "(locked)" in text-dim 11px until simulation reaches obs-2
4. "Water expansion" — checked
5. "D8 corridor" — checked
6. "OSM assets" — checked

**Divider** (1px border)

**Opacity slider:**
- "Opacity" label (12px text-dim)
- Range slider 0-100%, accent track, 100% default

**No elevation slices. No coordinate readout. No tile counts. No projection info.**

### Map Canvas (flex-1)

- Dark background (`#0a0f1e`)
- Basin polygon: accent outline, 2px, 8% fill
- Water expansion: blue fill, 40% opacity
- D8 corridor: amber dashed line, 2px
- Asset markers: 12px circles, colored by status (green/amber/red), 2px dark border
- **Step label** (top-left, inside map): panel background, 1px border, 12px text, shows "Obs 2 — 2026-08-04"

**No compass rose. No scale bar. No crosshair. No fullscreen button. No coordinate overlay.**

### Bottom Swipe Compare (below map, panel styled, 12px gap from map)

**Title:** "Before / After" (16px)

**Swipe area** (100px height):
- Left: dark background, "Before — 3.0 km²" label, 12px text-dim
- Right: blue-tinted background, "After — 4.1 km² (+14.3%)" label, 12px accent color
- Vertical drag handle: 2px accent line with a small grabber

**No confidence level. No "drag horizontally to inspect" hint. No sensor labels.**

### Right Dock — Legend + Asset Detail (240px, panel styled, 16px padding)

**Legend section:**
- "Legend" title (16px)
- 3 rows (10px dot + label, 13px, 8px spacing):
  - Green dot + "Safe"
  - Amber dot + "Buffered"
  - Red dot + "Inundated"

**Selected Asset card** (only when a marker is clicked):
- Asset name (16px, text color)
- "Type: bridge" (13px text-dim)
- "Distance: 60m from corridor" (13px text-dim)
- "Population: 1,240" (13px text-dim, only for villages)
- Status badge (BUFFERED / INUNDATED / SAFE)
- "Fly to" button (ghost, 12px) and "Review →" link (accent, 12px)

**No structural risk score. No peak flow ETA. No stream latency. No sensor IDs. No export buttons.**

---

## Page 3: Timeline View

The simulation controller and observation sequence. This is where the demo starts.

### Layout (vertical stack, 16px gap, 16px padding)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Timeline                                                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  [▶ Run Simulation]    0/3  ━━━━━━━━━━━━━━━━━━━━━━━━━                    │
├──────────────────────────────────────────────────────────────────────────┤
│  ★ 12 days of early warning between Obs 1 (+8%) and Obs 2 (+28%)         │
├──────────────────────────────────────────────────────────────────────────┤
│  Obs 2: 95% cloud → SAR path                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │Baseline │ │ Obs 1   │ │ Obs 2   │ │ Obs 3   │                        │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### View Title

"Timeline" (20px, text color)

### Simulation Controller (panel styled, full width, 16px padding)

- **"▶ Run Simulation"** button — primary, accent background, dark text, 14px
  - Becomes "Running..." (disabled) while running
  - Becomes "↻ Replay" after completion
- **Step counter:** "0/3" (14px text-dim)
- **Progress bar:** flex-1 width, 4px height, border track, accent fill

**No status text. No tick marks. No replay/scrub buttons.** The button + progress bar is enough.

### Prevention Callout (panel with 3px green left border, 16px padding, only after obs-2)

- "★" in green
- "12 days of early warning between Obs 1 (+8%) and Obs 2 (+28%)" (14px text, numbers in green)
- One line. No evacuation corridor badge. No verified-user icon.

### Router Strip (panel styled, full width, 12px padding, only for SAR-routed observations)

- "Obs 2:" (14px text-dim)
- "95% cloud" (14px text)
- "→" (14px accent)
- "SAR path" (14px text)
- One line. No cloud icon. No weather-adaptive routing explanation. The badge is enough.

### Observation Cards (horizontal row, 4 equal-width cards, 12px gap)

Each card is a panel (8px radius, 1px border, 16px padding):

**Active card:** accent border (2px)
**Past card:** normal border
**Future card:** 40% opacity, not clickable

**Card content (top to bottom, 8px spacing between items):**
1. **Label:** "Baseline" or "Obs 1" or "Obs 2" or "Obs 3" (14px, medium weight)
2. **Date:** "2025-11-22" (12px text-dim, monospace)
3. **Sensor:** "S2 Optical" or "S1 SAR" (12px, info color for optical, accent for SAR)
4. **Metrics** (4 rows, each: label left 12px text-dim, value right 14px text):
   - "cloud" → "5%" / "0%" / "95%"
   - "rain 24h" → "0mm" / "18mm" / "85mm"
   - "area" → "3.0 km²" / "3.2 km²" / "4.1 km²"
   - "change" → "+0%" / "+8%" / "+28%" (with ▲ arrow if positive)
5. **Severity badge** (bottom): SAFE / WATCH / ELEVATED / CRITICAL

**No trend arrows section. No "rain 7d" (redundant with 24h). No processing version. No quality score.**

---

## Page 4: Review View

The human decision console. The most important screen. Only shows when severity is elevated or critical.

### Layout (2-column + sticky bottom bar)

```
┌──────────────────────────────┬──────────────────────────────┐
│  EVIDENCE (left, 45%)        │  RISK & REASONS (right, 55%) │
│                              │                              │
│  Before / After images       │  Risk Scores                 │
│                              │    H  ████████░░  0.62       │
│                              │    E  ██████░░░░  0.48       │
│                              │    D  ████░░░░░░  0.31       │
│                              │    C  ███████░░░  0.76       │
│                              │    [ELEVATED]                │
│                              │                              │
│                              │  Evidence Reasons (8)        │
│                              │    1. temporal trend...      │
│                              │    2. water expansion...     │
│                              │    3. rainfall...            │
│                              │    ...                       │
├──────────────────────────────┴──────────────────────────────┤
│  DISEASE ACTIONS          │  EXPOSED ASSETS                 │
│  Well 3: submerged         │  #  Asset       Type   Status  │
│    → chlorine ×200         │  1  Well 3      well   INUND   │
│    → boil water notice     │  2  Bridge      bridge BUF     │
│  7-day surveillance        │  3  Road 4      road   BUF     │
│                            │  4  Chhukung    village BUF    │
├──────────────────────────────────────────────────────────────┤
│  [✓ Confirm]  [✗ Reject]  [⏸ Postpone]     reviewer: coord  │
└──────────────────────────────────────────────────────────────┘
```

### View Title

"Review — Run run-0004" (20px, "run-0004" in monospace accent color)

### Evidence Panel (left, 45%, panel styled, 16px padding)

**Title:** "Evidence" (16px)

**Two raster preview boxes** (side by side, 140px height, 8px radius):
- Left: dark gradient, "Before" label (12px text-dim), "3.0 km²" (14px text)
- Right: blue-tinted gradient, "After" label (12px text-dim), "4.1 km²" (14px accent)

**File path** (below images, 12px monospace text-dim):
- "data/processed/obs-003_expansion_mask.tif"

### Risk Scores Panel (right top, panel styled, 16px padding)

**Title:** "Risk Scores" (16px)

**4 gauge rows** (each: label + bar + value, 12px spacing):
- "H" (14px, danger color) → danger bar (height 6px, 4px radius) → "0.62" (24px monospace)
- "E" (14px, warn color) → warn bar → "0.48" (24px monospace)
- "D" (14px, info color) → info bar → "0.31" (24px monospace)
- "C" (14px, safe color) → safe bar → "0.76" (24px monospace)

**Severity badge** below gauges: "ELEVATED" (orange)

### Evidence Reasons Panel (right middle, panel styled, 16px padding)

**Title:** "Evidence Reasons (8)" (16px)

**8 reason rows** (numbered, 13px, 8px spacing):
1. "temporal trend 'rapidly' contributes 0.3×0.90 to H"
2. "water-area expansion +14.3% contributes 0.25×0.48 to H"
3. "rainfall 24h 0mm / 7d 0mm contributes 0.2×0.00 to H"
4. "terrain slope 31° contributes 0.15×0.69 to H"
5. "downstream proximity contributes 0.1×1.00 to H"
6. "hazard H=0.59 scales exposure directly"
7. "exposed population 1240 → vulnerability 0.62"
8. "1 inundated water point → disease factor 0.20"

Number in accent color, medium weight. Text in normal text color.

### Disease Actions Panel (bottom left, panel styled, 16px padding)

**Title:** "Disease Prevention" (16px)

**Per-well blocks** (12px spacing):
- "Well 3 — submerged" (13px, danger color)
  - "→ chlorine ×200 tablets" (13px text)
  - "→ boil water notice" (13px text)
- "Well 7 — encircled" (13px, warn color)
  - "→ monitor for contamination" (13px text)

**General actions:**
- "→ 7-day diarrheal surveillance" (13px text)
- "→ alternate water supply for 1,240 people" (13px text)

**No icons on action items. No colored dots.** The text and color coding on the well header is enough.

### Exposed Assets Table (bottom right, panel styled, 16px padding)

**Title:** "Exposed Assets" (16px)

**Table** (13px, 12px header):
| # | Asset | Type | Dist | Status |
|---|---|---|---|---|
| 1 | Well 3 | well | 90m | INUNDATED |
| 2 | Hillary Bridge | bridge | 60m | BUFFERED |
| 3 | Road 4 | road | 40m | BUFFERED |
| 4 | Chhukung | village | 210m | BUFFERED |

- Header: text-dim, 12px
- Rows clickable (selects asset, jumps to Map)
- Status as text badges (no filled backgrounds)

### Decision Bar (sticky bottom, full width, 52px, border-top 1px, panel background)

**3 states:**

**Default:**
- "✓ Confirm" — primary button, safe (green) background, dark text, 14px
- "✗ Reject" — danger button, danger background, white text, 14px
- "⏸ Postpone" — warn button, warn background, dark text, 14px
- "reviewer: coordinator-01" — right-aligned, 12px text-dim

**Confirm step (after clicking Confirm):**
- "Confirm SOS dispatch?" — 14px text
- "Yes, confirm" — primary button, green
- "Cancel" — ghost button

**Locked (after any decision):**
- "✓ Confirmed" or "✗ Rejected" or "⏸ Postponed" — 14px, in decision color
- "Send Dispatch" button (ghost, 12px) — only when confirmed

---

## Page 5: Audit View

The dispatch record and audit trail. Clean, factual, no blockchain aesthetic.

### Layout (vertical stack, 16px gap, 16px padding)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Audit                                                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  Compressed Payload                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │ {"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,...}             ││
│  │                                                                      ││
│  │  118 / 250 bytes  ████████████░░░░░░░  ✓ LoRa-compatible            ││
│  └──────────────────────────────────────────────────────────────────────┘│
│  [Preview]  [Copy]                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│  Channels                                                                 │
│  [SMS]  ✓ delivered    [LoRa]  ✓ delivered    [Satellite]  ⏳ queued     │
├──────────────────────────────────────────────────────────────────────────┤
│  Audit Trail                                                              │
│  ┌────────────────────────┬────────────────┬──────────┬───────────────┐  │
│  │ Time (UTC)             │ Actor          │ Action   │ Detail        │  │
│  ├────────────────────────┼────────────────┼──────────┼───────────────┤  │
│  │ 2026-09-05T04:10:33Z   │ pipeline       │ run      │ {"run_id":..} │  │
│  │ 2026-09-05T04:17:15Z   │ coordinator-01 │ review   │ {"decision":.}│  │
│  │ 2026-09-05T04:17:15Z   │ coordinator-01 │ dispatch │ {"channel":.} │  │
│  └────────────────────────┴────────────────┴──────────┴───────────────┘  │
│  Append-only — no edits, no deletes.                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

### View Title

"Audit" (20px, text color)

### Payload Card (panel styled, 16px padding)

**Title row:**
- "Compressed Payload" (16px)
- "Preview" button (ghost, 12px) and "Copy" button (ghost, 12px), right-aligned

**Payload box** (monospace, 14px, darkest background `#0a0f1e`, 8px radius, 12px padding):
```
{"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,"exp_pop":1240,"crit":["BR-12","RD-4"],"med_act":"BOIL_WATER_NOW"}
```

**Metadata row** (12px spacing below payload):
- Byte meter: 200px bar, 4px height, border track, safe fill
- "118 / 250 bytes" (12px monospace, safe color)
- "✓ LoRa-compatible" (12px, safe color)

**No hex prefix. No encoding info. No alert_id inline. No frame payload ratio label.**

### Channel Simulator (panel styled, 16px padding)

**Title:** "Channels" (16px)

**3 channel rows** (horizontal, 16px gap):
- "SMS" label + "✓ delivered" (12px, safe color)
- "LoRa" label + "✓ delivered" (12px, safe color)
- "Satellite" label + "⏳ queued" (12px, warn color)

Each channel is a small panel (border, 8px padding) with the channel name and status below it.

**No geofence subtitle. No recipient group count.** The channel names and statuses are the content.

### Audit Trail (panel styled, 16px padding)

**Title:** "Audit Trail" (16px)

**Table** (13px, monospace for time and detail columns):
| Time (UTC) | Actor | Action | Detail |
|---|---|---|---|
| 2026-09-05T04:10:33Z | pipeline | run | {"run_id":"run-0004",...} |
| 2026-09-05T04:17:15Z | coordinator-01 | review | {"decision":"confirm",...} |
| 2026-09-05T04:17:15Z | coordinator-01 | dispatch | {"channel":"sms",...} |

- Header: text-dim, 12px
- Action column: text badge (run=warn color, review=info color, dispatch=accent color)
- Detail column: 12px monospace, text-dim, truncated with ellipsis

**Footer note** (12px text-dim):
- "Append-only — no edits, no deletes."

**No PRD section references. No schema trigger mentions. No chain IDs. No block numbers.**

### Transmission Preview Modal (overlay, triggered by "Preview")

**Overlay:** `rgba(0,0,0,0.5)` background, click to close

**Modal** (centered, 440px max-width, panel background, 8px radius, 16px padding):
- "×" close button (top-right, 14px text-dim)
- "Transmission Preview" (16px title)
- Dark message box (`#0a0f1e`, 8px radius, 12px padding):
  - "HIGH ALERT — GLOF_FL" (16px, danger color, medium weight)
  - "Alert: siren-04" (14px)
  - "Sector: B" (14px)
  - "Population: 1,240" (14px)
  - "Assets: BR-12, RD-4" (14px)
  - "Action: BOIL_WATER_NOW" (14px, warn color)
- "Decoded view of the 118-byte payload." (12px text-dim)

---

## Empty States

### Review — No Alerts

Centered in the view area:
- "No alerts requiring review" (16px text)
- "Run the simulation from Timeline to generate observations." (14px text-dim)

No large icon. No checkmark. Just text.

### Audit — No Dispatches

Centered:
- "No dispatches yet" (16px text)
- "Confirm a review and send a dispatch to see the audit trail." (14px text-dim)

---

## What NOT to Include

These elements are intentionally excluded to keep the interface clean:

- No session IDs, chain IDs, block numbers, or blockchain aesthetic
- No hex prefixes on payloads
- No orbital synch badges, gate numbers, or mode indicators
- No compass roses, scale bars, or coordinate readouts on the map
- No elevation slice buttons or tile counts
- No stream latency, downstream sensor IDs, or telemetry capsules
- No structural risk scores or peak flow ETAs on asset cards
- No confidence level percentages on swipe compare
- No "EXPORT KML" or export buttons
- No tick marks under the timeline controller
- No glow effects, no pulse animations (except the alert banner slide-in)
- No decorative gradients (solid colors only)
- No Material Symbols icons (use text labels and simple ASCII markers instead)
- No uppercase tracked-out label-caps typography
- No fake technical jargon that isn't backed by real data

---

## Key Interactions

1. **Run Simulation** on Timeline → progress bar fills, cards activate in sequence, severity escalates
2. **Alert banner** appears when severity reaches elevated → click navigates to Review
3. **Before/After swipe** on Map → drag to reveal water expansion
4. **Confirm** on Review → two-step confirm → decision locks → "Send Dispatch" appears
5. **Channel buttons** on Audit → click to simulate delivery
6. **Copy** on Audit → copies payload, shows "Copied" feedback
7. **Preview** on Audit → modal shows decoded recipient-facing message
8. **Asset click** on Map → detail card appears → "Review →" jumps to Review
9. **Asset row click** on Review → jumps to Map with asset selected
