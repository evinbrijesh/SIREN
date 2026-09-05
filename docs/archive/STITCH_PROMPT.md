# Stitch Design Prompt — SIREN Coordinator Console

> **ARCHIVED DESIGN INPUT — not spec.** This prompt was used to generate early UI mockups during the design phase. The numbers below (e.g., obs-003 as 2026-09-04, +14.3%, Elevated) are **stale** and do not match the implemented canonical scenario (obs-003 = 2026-08-12, +43%, Critical, S1 SAR). Refer to `docs/spec/PRD.md` §16 and `README.md` for the current demo scenario. The implemented UI is a Tailwind operational console — see `docs/design/UI_DESIGN.md` for the current design spec.

## Project Overview

Design a **dark-themed emergency operations console** called SIREN (Satellite-Informed Risk & Emergency Network). It is a disaster-response decision platform for Himalayan glacial basins. An emergency coordinator uses it to monitor satellite observations of a glacial lake, review escalating hazard evidence, confirm alerts, and dispatch compressed warnings to downstream communities.

The app has **4 main views** (Map, Timeline, Review, Audit) plus a **shared app shell** with a navigation bar and alert banner. It is a **single-page application** — all views share the same shell, switched via tabs.

**Vibe:** Mission control. Dark, high-contrast, information-dense. This is a tool someone uses during a disaster, not a marketing site. Think Bloomberg Terminal meets emergency dispatch console.

---

## Design System

### Color Palette (dark theme)

| Token | Hex | Usage |
|---|---|---|
| Background | `#0F172A` | App background (slate-900) |
| Panel | `#1E293B` | Cards, panels, docks (slate-800) |
| Panel-2 | `#334155` | Nested panels, borders, dividers (slate-700) |
| Text | `#E2E8F0` | Primary text (slate-200) |
| Text-dim | `#94A3B8` | Secondary/label text (slate-400) |
| Accent | `#06B6D4` | SIREN brand color, active states, cyan |
| Safe | `#22C55E` | Green — safe assets, success, confirmed |
| Warn | `#F59E0B` | Amber — advisory, buffered risk, postponed |
| Elevated | `#F97316` | Orange — elevated severity |
| Danger | `#EF4444` | Red — critical, inundated, rejected |
| Info | `#3B82F6` | Blue — informational, optical sensor |

### Typography

- **UI font:** Inter (or system sans-serif)
- **Monospace font:** JetBrains Mono (or SF Mono) — used for telemetry data, JSON payloads, timestamps, coordinates
- **Sizes:** 12px (labels/badges), 14px (body), 16px (card titles), 20px (view titles), 28px (gauge values), 48px (nav brand)

### Spacing & Radius

- 8px grid. Panel padding 16px. Card gap 12px.
- Border radius: 8px (panels/cards), 4px (badges/buttons), 50% (status dots)

### Status Chips/Badges

- `informational` → blue outline badge
- `watch` → amber outline badge
- `elevated` → orange filled badge
- `critical` → red filled badge with subtle pulse glow
- `safe` → green outline badge
- `confirmed` → green filled badge
- `rejected` → red filled badge

---

## Page 1: App Shell (shared across all views)

### Navigation Bar (48px height, full width)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SIREN   Dudh Koshi/Imja ▾    ● LIVE    [Map] [Timeline] [Review] [Audit]    │
│  (brand)  (basin selector)    (status)   (4 tab buttons with 1-4 key hints)   │
│                                          ● OFFLINE — ALL SYSTEMS LOCAL    ⟲  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Left section:**
- "SIREN" brand text in cyan accent color, bold, 20px
- "Dudh Koshi/Imja" basin selector with dropdown arrow, 14px text-dim
- "● LIVE" status indicator with a small green pulsing dot

**Center section:**
- 4 tab buttons: Map, Timeline, Review, Audit
- Active tab: cyan accent text with a 2px cyan bottom border
- Inactive tabs: text-dim color
- Each tab has a small number hint (1-4) in 10px text-dim to the right

**Right section:**
- "● OFFLINE — ALL SYSTEMS LOCAL" badge (only visible when offline) — amber text, amber outline, 12px
- "RESET ⟲" button — ghost button, 12px, text-dim, resets the console

### Alert Banner (40px height, slides in below nav, only when severity is elevated or critical)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ⚠  ELEVATED  —  +14.3% water expansion → Review                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

- Red/orange left border (4px, matching severity color)
- "⚠" warning icon
- Severity text in uppercase, bold, severity color
- Brief description: expansion percentage + "→ Review" call to action
- Clicking the banner navigates to the Review tab
- Background: severity color at 10% opacity
- Slides in from top with animation

### Provenance Footer (24px height, bottom of screen)

```
Sentinel-2 · Sentinel-1 · SRTM · Open-Meteo · © OSM · pipeline v0.1.0
```

- Small text (11px), text-dim color, centered
- Data source attribution
- Separator dots between sources

### Toast Notifications (floating, bottom-right)

- Appears 16px from bottom-right edge
- 3 types: error (red left border), info (blue), success (green)
- Auto-dismisses after 4 seconds
- Click to dismiss
- 14px text, panel background, 8px radius

---

## Page 2: Map View (default landing view)

This is the "hero" view — a full-bleed interactive map with side docks.

### Layout (3-column flex)

```
┌──────────────┬────────────────────────────────────────┬─────────────────┐
│ LEFT DOCK    │  MAP CANVAS (full-bleed)               │ RIGHT DOCK      │
│ 220px        │                                        │ 280px           │
│              │  ┌──────────────────────────────────┐  │                 │
│ LAYERS       │  │ Step badge: "Obs 2 — 2026-08-04"│  │ ASSET LEGEND    │
│ ☑ Basin AOI  │  └──────────────────────────────────┘  │ ● Safe          │
│ ☐ DEM hill   │                                        │ ● Buffered      │
│ ☑ Optical    │  (MapLibre GL canvas rendering:        │ ● Inundated     │
│ ☐ SAR ⚡     │   - basin polygon outline (cyan)       │                 │
│ ☑ Water exp  │   - water expansion fill (blue)        │ ─────────────── │
│ ☑ D8 corridor│   - D8 corridor line (amber dashed)    │                 │
│ ☑ OSM assets │   - asset markers (green/amber/red)    │ SELECTED ASSET  │
│              │   - dark background (#0a0f1e))         │ (detail card)   │
│ ───────────  │                                        │                 │
│ Opacity      │                                        │                 │
│ ━━━━●━━━━━━  │                                        │                 │
│              │  ┌──────────────────────────────────┐  │                 │
│              │  │ BEFORE / AFTER SWIPE             │  │                 │
│              │  │  drag handle ←→ reveals change   │  │                 │
│              │  └──────────────────────────────────┘  │                 │
└──────────────┴────────────────────────────────────────┴─────────────────┘
```

### Left Dock — Layer Toggle Panel (220px wide, card styled)

**Title:** "Layers" (16px card title)

**7 toggle rows**, each with a checkbox + label (13px):
1. "Basin AOI" — checked by default
2. "DEM hillshade" — unchecked
3. "Optical baseline" — checked
4. "SAR backscatter" — unchecked, locked until simulation reaches obs-2. When locked: dimmed checkbox, "(locked)" label in text-dim. When revealed: "⚡ revealed" in cyan accent
5. "Water expansion" — checked
6. "D8 + OSM corridor" — checked
7. "OSM assets" — checked

**Divider line** (1px panel-2)

**Opacity slider section:**
- "Opacity fallback" label (12px text-dim)
- Range slider (0-100%), cyan accent track

### Map Canvas (flex-1, fills remaining space)

- Dark background (#0a0f1e)
- Basin polygon: cyan outline (#06B6D4), 2px, with 8% cyan fill
- Water expansion: blue fill (#3B82F6), 40% opacity
- D8 corridor: amber dashed line (#F59E0B), 3px, dasharray [2,1]
- Asset markers: 14px circles, colored by status (green/amber/red), 2px dark border
- **Step badge** (top-left overlay): semi-transparent dark background (rgba(15,23,42,0.85)), 1px panel-2 border, 12px cyan text, shows current observation label

### Bottom Swipe Compare (below map, card styled)

**Title:** "Before / After Swipe — water expansion" (16px)

**Swipe area** (120px height):
- Left side: dark gradient (#1a2a4a → #0a0f1e), "BEFORE (baseline)" label top-left, 11px text-dim
- Right side: blue-tinted gradient (#1a3a5a → #0a1f3e), "AFTER" label top-right, 11px blue
- Vertical drag handle at swipe position (2px cyan line with a grabber circle)
- Dragging left/right reveals more of the "after" layer

### Right Dock — Legend + Asset Detail (280px wide)

**Asset Legend card:**
- Title: "Asset Legend" (16px)
- 3 rows, each with a 10px colored dot + label (13px):
  - Green dot + "Safe"
  - Amber dot + "Buffered (within 100m)"
  - Red dot + "Inundated"

**Selected Asset card** (only when an asset marker is clicked on the map):
- Title: asset name (16px) — e.g. "Hillary Bridge"
- "Type: bridge" (13px text-dim)
- "Buffer: ±75 m" (13px text-dim)
- "60 m from corridor" (13px text-dim)
- "Population: 1240" (13px text-dim, only for villages)
- Status badge: "BUFFERED" (amber) or "INUNDATED" (red) or "SAFE" (green)
- Two buttons (ghost style, 12px):
  - "Fly to" — zooms map to asset
  - "Review →" — navigates to Review tab

---

## Page 3: Timeline View

Shows the observation sequence and the simulation controller. This is where the demo's "Run Simulation" button lives.

### Layout (vertical stack)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Timeline — Observation Sequence                              (view title) │
├──────────────────────────────────────────────────────────────────────────────┤
│  SIMULATION CONTROLLER                                                       │
│  [▶ Run Simulation]  status: before → disaster  ━━━━●━━━━  0/3              │
├──────────────────────────────────────────────────────────────────────────────┤
│  ★ PREVENTION CALLOUT (green-bordered card, appears after obs-2)            │
│  Obs 1 flagged +8% on 2026-07-23 — 12 days of warning before the +28% surge │
├──────────────────────────────────────────────────────────────────────────────┤
│  WEATHER-ADAPTIVE ROUTER STRIP                                               │
│  Obs 2: Optical cloud 95% → ⚡ SWITCHED TO SAR PATH    [cyan badge]         │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│  │Baseline │ │ Obs 1   │ │ Obs 2   │ │ Obs 3   │  ← horizontal cards       │
│  │2025-11  │ │ 2026-07 │ │ 2026-08 │ │ 2026-09 │                           │
│  │S2 Opt   │ │S1 SAR   │ │S1 SAR   │ │S2 Opt   │                           │
│  │cloud 5% │ │cloud 0% │ │cloud 95%│ │cloud 10%│                           │
│  │rain 0   │ │rain 18  │ │rain 85  │ │rain 0   │                           │
│  │3.0 km²  │ │3.2 km²  │ │4.1 km²  │ │3.4 km²  │                           │
│  │→ +0%    │ │▲ +8%    │ │▲ +28%   │ │▲ +14.3% │                           │
│  │[SAFE]   │ │[WATCH]  │ │[CRIT]   │ │[ELEV]   │                           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Simulation Controller (card styled, full width)

- **"▶ Run Simulation"** button — primary button, cyan background, white text, 14px
  - Changes to "▶ Running..." (disabled) while running
  - Changes to "⟲ Replay" after completion
- **Status text:** "status: before → disaster ·" (14px text-dim)
- **Progress bar:** 200px wide, 6px tall, panel-2 track, cyan fill, animated width
- **Step counter:** "0/3" (12px text-dim)

### Prevention Callout (card with green border, appears after obs-2 is reached)

- Green left border (4px)
- "★ PREVENTION:" prefix in green
- Text: "Obs 1 flagged +8% on 2026-07-23 — **12 days of warning** before the +28% surge. Lead time SIREN would have bought."
- 14px text, green color for key numbers

### Weather-Adaptive Router Strip (card styled, full width)

- "Obs 2:" label (text-dim)
- "Optical cloud 95%" text
- "→" arrow in cyan accent
- "⚡ SWITCHED TO SAR PATH" badge — cyan background at 20%, cyan border, cyan text, 12px
- Right-aligned: "Weather-adaptive routing: SAR is all-weather, effective cloud 0.0%" (12px text-dim)

### Observation Cards (horizontal row, 4 cards, equal width)

Each card is a clickable panel (card styled) with:

**Active card:** cyan border (2px), slightly elevated
**Past card:** normal styling
**Future card (not yet reached):** 50% opacity, greyed out, not clickable

**Card content (top to bottom):**
1. **Date label:** "Baseline — 2025-11-22" or "Obs 1 — 2026-07-23" (14px, bold)
2. **Sensor badge:** "S2 Optical" (blue info badge) or "S1 SAR" (cyan accent badge), 12px
3. **Metric rows** (each: label left in text-dim 12px, value right in text 14px):
   - "cloud" → "5%" or "0%" or "95%"
   - "rain 24h" → "0 mm" or "18 mm" or "85 mm"
   - "rain 7d" → "0 mm" or "64 mm" or "192 mm"
   - "area" → "3.0 km²" or "3.2 km²" or "4.1 km²"
   - "change" → "→ +0.0%" or "▲ +8.0%" or "▲ +28.0%" (trend arrow + colored value)
4. **Severity badge** (bottom, full width):
   - "INFORMATIONAL" → blue outline
   - "WATCH" → amber outline
   - "ELEVATED" → orange filled
   - "CRITICAL" → red filled with subtle pulse

---

## Page 4: Review View

The human-in-the-loop decision console. This is the most important view — where the coordinator decides whether to dispatch an alert. Only shows content when severity is elevated or critical.

### Layout (3-column + sticky bottom bar)

```
┌──────────────────────┬──────────────────────────────┬─────────────────┐
│  EVIDENCE PANEL      │  RISK GAUGE + REASONS        │  RIGHT DOCK     │
│  (40% width)         │  (flex-1)                    │  (320px)        │
│                      │                              │                 │
│  Before/After        │  ┌────────────────────────┐  │  DISEASE        │
│  raster images       │  │  H  ████████░░  0.62   │  │  PREVENTION     │
│  side by side        │  │  E  ██████░░░░  0.48   │  │  ACTION SHEET   │
│                      │  │  D  ████░░░░░░  0.31   │  │                 │
│  Change mask path    │  │  conf ███████░░  0.76  │  │  Well 3:        │
│  (monospace)         │  │                        │  │   submerged     │
│                      │  │  [ELEVATED badge]      │  │  → chlorine     │
│                      │  └────────────────────────┘  │  → BOIL WATER   │
│                      │                              │                 │
│                      │  EVIDENCE REASONS (8)        │  Well 7:        │
│                      │  1. temporal trend 'rapidly' │  encircled      │
│                      │     contributes 0.3*0.90     │  → monitor      │
│                      │  2. water-area expansion     │                 │
│                      │     +14.3% contributes...    │  7-day disease  │
│                      │  3. rainfall 24h 0mm...      │  surveillance   │
│                      │  4. terrain slope 31°...     │                 │
│                      │  5. downstream proximity...  │  ─────────────  │
│                      │  6. hazard H=0.59 scales...  │                 │
│                      │  7. exposed population...    │  EXPOSED ASSETS │
│                      │  8. 1 inundated water point  │  (ranked table) │
│                      │                              │  # Asset Type   │
│                      │                              │  1 Well3 well   │
│                      │                              │  2 BR-12 bridge │
│                      │                              │  3 RD-4  road   │
│                      │                              │  4 Vill2 village│
├──────────────────────┴──────────────────────────────┴─────────────────┤
│  DECISION BAR (sticky bottom, full width)                              │
│  [✓ Confirm SOS]  [✗ Reject]  [⏸ Postpone]    reviewer: coordinator-01│
└────────────────────────────────────────────────────────────────────────┘
```

### View Title

"Review — Run run-0004" (20px, top of view)

### Evidence Panel (left, 40% width, card styled)

**Title:** "Evidence — Before / After" (16px)

**Two raster preview boxes** (side by side, 160px height each):
- Left: dark gradient (#1a2a4a → #0a0f1e), "BEFORE" label centered, 12px text-dim
- Right: blue-tinted gradient (#1a3a5a → #0a1f3e), "AFTER (+14.3%)" label centered, 12px blue

**Change mask path** (below images):
- "Change mask: data/processed/obs-003_expansion_mask.tif" (13px text-dim, monospace)

### Risk Gauge Panel (center, card styled)

**Title:** "Risk Scores" (16px)

**4 gauge rows** (each row: label + horizontal bar + value):
- "H" label (14px bold, red) → red bar fill (width = value×100%) → "0.62" (28px monospace)
- "E" label (14px bold, amber) → amber bar fill → "0.48" (28px monospace)
- "D" label (14px bold, blue) → blue bar fill → "0.31" (28px monospace)
- "conf" label (14px bold, green) → green bar fill → "0.76" (28px monospace)

Gauge bar: panel-2 track (6px height, 4px radius), colored fill

**Severity badge** (below gauges):
- "ELEVATED" → orange filled badge, uppercase, bold

### Evidence Reasons Panel (center, below gauges, card styled)

**Title:** "Evidence Reasons (8)" (16px)

**8 reason rows** (each: numbered prefix + text, 13px):
1. "temporal trend 'rapidly' contributes 0.3*0.90 to H"
2. "water-area expansion +14.3% contributes 0.25*0.48 to H"
3. "rainfall 24h 0.0mm / 7d 0.0mm contributes 0.2*0.00 to H"
4. "terrain slope 31.0° contributes 0.15*0.69 to H"
5. "downstream proximity on drainage contributes 0.1*1.00 to H"
6. "hazard H=0.59 scales exposure directly"
7. "exposed population 1240 → vulnerability factor 0.62"
8. "1 inundated/encircled water points → factor 0.20"

Number prefix in cyan accent, bold. Reason text in normal text color.

### Right Dock — Disease Prevention Action Sheet (320px, card styled)

**Title:** "Disease Prevention Actions" (16px)

**Per-well action blocks** (each in a tinted mini-panel):

For submerged wells (red-tinted background rgba(239,68,68,0.08)):
- "Well 3: submerged" (13px bold)
- "→ chlorine ×200 tablets" (13px, red dot prefix)
- "→ BOIL WATER NOW notice" (13px, red dot prefix)

For encircled wells (amber-tinted background rgba(245,158,11,0.08)):
- "Well 7: encircled" (13px bold)
- "→ monitor for contamination" (13px, amber dot prefix)

**General actions:**
- "7-day diarrheal disease surveillance window" (13px, blue dot prefix)
- "Safe water sources: identify alternate supply for 1240 people" (13px, green dot prefix)

### Right Dock — Exposed Assets Table (below action sheet, card styled)

**Title:** "Exposed Assets (ranked by distance)" (16px)

**Table** (compact, 13px):
| # | Asset | Type | Dist | Status |
|---|---|---|---|---|
| 1 | Well 3 | well | 90m | INUNDATED (red badge) |
| 2 | Hillary Bridge | bridge | 60m | BUFFERED (amber badge) |
| 3 | Road 4 | road | 40m | BUFFERED (amber badge) |
| 4 | Chhukung | village | 210m | BUFFERED (amber badge) |

- Rows are clickable — clicking selects the asset and jumps to Map view
- Header row: text-dim, 12px
- Body rows: 13px, alternating subtle background

### Decision Bar (sticky bottom, full width, 56px height)

**3 states:**

**Default state (no decision yet):**
- "✓ Confirm SOS" — primary button, green background, white text, 14px
- "✗ Reject" — danger button, red background, white text, 14px
- "⏸ Postpone" — warn button, amber background, white text, 14px
- "reviewer: coordinator-01" — right-aligned, 12px text-dim

**Confirm step (after clicking Confirm SOS):**
- "Confirm SOS dispatch?" — bold red text, 14px
- "Yes, confirm" — primary button, green
- "Cancel" — ghost button

**Decision locked (after any decision is made):**
- "✓ Decision recorded: CONFIRM" — green text with checkmark
- "📤 Send Dispatch" — ghost button (only when confirmed)
- Background tinted with decision color (green for confirm, red for reject, amber for postpone)

---

## Page 5: Audit View

Shows the dispatch payload, channel simulator, and immutable audit trail. This is the "accountability" view.

### Layout (vertical stack)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Audit — Lineage & Resilient Alerting                        (view title)    │
├──────────────────────────────────────────────────────────────────────────────┤
│  COMPRESSED PAYLOAD (card)                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ {"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,"exp_pop":1240,..}│  │
│  │                                                                        │  │
│  │  [118 / 250 bytes]  ████████████░░░░░░░  ✓ LoRa-compatible            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  [👁 Preview]  [⧉ Copy]                                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  CHANNEL SIMULATOR (card)                                                    │
│  → sector-b geofence · 3 recipient groups                                   │
│  [SMS] ✓delivered   [LoRa Mesh] ✓delivered   [Satellite] ⏳queued           │
├──────────────────────────────────────────────────────────────────────────────┤
│  AUDIT TRAIL — IMMUTABLE LINEAGE (card)                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Timestamp (UTC)     │ Actor          │ Action   │ Detail (JSON)       │  │
│  ├─────────────────────┼────────────────┼──────────┼─────────────────────┤  │
│  │ 2026-09-05T04:10:33 │ pipeline       │ run      │ {"run_id":"run-004"}│  │
│  │ 2026-09-05T04:17:15 │ coordinator-01 │ review   │ {"decision":"confir"}│  │
│  │ 2026-09-05T04:17:15 │ coordinator-01 │ dispatch │ {"channel":"sms"}   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│  ✓ Append-only — UPDATE and DELETE blocked by schema triggers               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Compressed Payload Card

**Title row:**
- "Compressed Payload (Track 7.ii)" (16px card title)
- Two buttons (right-aligned, ghost style, 12px):
  - "👁 Preview" — opens a modal showing the decoded recipient-facing message
  - "⧉ Copy" — copies payload to clipboard, changes to "✓ Copied" for 2 seconds

**Payload box** (monospace, dark background #0a0f1e, 8px radius, 16px padding):
```json
{"aid":"siren-04","sec":"B","haz":"GLOF_FL","lvl":3,"exp_pop":1240,"crit":["BR-12","RD-4"],"med_act":"BOIL_WATER_NOW"}
```
- 14px monospace, text color, word-break all

**Payload metadata row:**
- **Byte meter:** horizontal bar (200px, 6px height, panel-2 track)
  - Green fill (width = bytes/250 × 100%)
  - "118 / 250 bytes" label (12px monospace, green "ok" class)
- "✓ LoRa-compatible" badge (green outline)
- "alert_id: alert-8896" (12px text-dim)

### Channel Simulator Card

**Title:** "Channel Simulator" (16px)

**Subtitle:** "→ sector-b geofence · 3 recipient groups" (13px text-dim)

**3 channel buttons** (horizontal row, each with button + status badge below):
- "SMS" button → status: "✓ delivered" (green badge)
- "LoRa Mesh" button → status: "✓ delivered" (green badge)
- "Satellite" button → status: "⏳ queued" (amber badge)

Buttons: primary style when idle, ghost style after dispatch. 12px text, 6px×12px padding.

**Human gate warning** (only when no review confirmed):
- "⚠ Dispatch blocked by human gate — confirm a review first." (12px amber)

### Audit Trail Card

**Title:** "Audit Trail — Immutable Lineage (alert: alert-0091)" (16px)

**Table** (compact, monospace for timestamps and JSON):
| Column | Width | Font | Size |
|---|---|---|---|
| Timestamp (UTC) | auto | monospace | 12px |
| Actor | auto | sans | 13px |
| Action | auto | badge | 12px |
| Detail (JSON) | 400px max | monospace | 12px text-dim |

**Action badges:**
- "run" → amber badge
- "review" → blue info badge
- "dispatch" → cyan accent badge

**Footer note** (below table):
- "✓ Append-only — UPDATE and DELETE blocked by schema triggers (PRD §7.8). No edit/delete affordances exist." (12px text-dim)

### Transmission Preview Modal (overlay, triggered by "Preview" button)

**Overlay:** semi-transparent dark background (rgba(0,0,0,0.6)), click to close

**Modal card** (centered, 500px max-width, panel background, 8px radius):
- "×" close button (top-right, ghost)
- "Transmission Preview — as recipient sees it" (16px title)
- Dark message box (#0a0f1e background, 8px radius):
  - "⚠ HIGH ALERT — GLOF_FL" (16px bold red)
  - "Alert ID: siren-04" (14px)
  - "Sector: B" (14px)
  - "Exposed population: 1240" (14px)
  - "Critical assets at risk: BR-12, RD-4" (14px)
  - "Medical action: BOIL_WATER_NOW" (14px amber)
- Footer: "This is the decoded view of the 118-byte compressed payload..." (12px text-dim)

---

## Empty States

### Review View — No Alerts

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                          ✓                                       │
│                  No alerts requiring review                      │
│        Run the simulation from the Timeline tab to               │
│          generate observations.                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

- Large green checkmark icon (48px)
- "No alerts requiring review" (16px text)
- "Run the simulation from the Timeline tab to generate observations." (14px text-dim)
- Centered in panel

### Audit View — No Dispatches

- "📋" icon (48px)
- "No dispatches yet" (16px)
- "Confirm a review and send a dispatch to see the audit trail." (14px text-dim)

---

## Keyboard Shortcuts (visual hint in nav)

- `1` → Map tab
- `2` → Timeline tab
- `3` → Review tab
- `4` → Audit tab
- `R` → Run simulation (from Timeline)
- `Esc` → Close toast/modal

---

## Key Interactions to Show

1. **Run Simulation button** on Timeline → advances through 3 observations with progress bar, each card activates in sequence, severity escalates from WATCH → CRITICAL → ELEVATED
2. **Alert banner** slides in when severity reaches elevated/critical → clicking it opens Review
3. **Before/After swipe** on Map → drag handle reveals water expansion
4. **Confirm SOS** on Review → two-step confirmation → decision locks → "Send Dispatch" button appears
5. **Channel simulator** on Audit → click SMS/LoRa/Satellite → status badges animate from idle → sent → delivered
6. **Payload copy** on Audit → click copy → "✓ Copied" feedback
7. **Transmission preview** modal on Audit → decoded recipient-facing message

---

## Technical Notes

- The app is a React SPA with MapLibre GL JS for the map
- All data comes from a FastAPI backend (SQLite + geospatial Python)
- The app must work fully offline (no external API calls at runtime)
- The map uses a dark style with no external tile server (solid color background)
- Status colors are semantic — never decorative
- Every score must show its evidence reasons (minimum 3 for elevated+)
- The dispatch payload must be ≤250 bytes (shown with a byte meter)
