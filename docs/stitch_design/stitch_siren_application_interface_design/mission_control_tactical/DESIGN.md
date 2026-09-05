---
name: Mission Control Tactical
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#bcc9cd'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#869397'
  outline-variant: '#3d494c'
  surface-tint: '#4cd7f6'
  primary: '#4cd7f6'
  on-primary: '#003640'
  primary-container: '#06b6d4'
  on-primary-container: '#00424f'
  inverse-primary: '#00687a'
  secondary: '#adc6ff'
  on-secondary: '#002e6a'
  secondary-container: '#0566d9'
  on-secondary-container: '#e6ecff'
  tertiary: '#ffb690'
  on-tertiary: '#552100'
  tertiary-container: '#ff853c'
  on-tertiary-container: '#672a00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#acedff'
  primary-fixed-dim: '#4cd7f6'
  on-primary-fixed: '#001f26'
  on-primary-fixed-variant: '#004e5c'
  secondary-fixed: '#d8e2ff'
  secondary-fixed-dim: '#adc6ff'
  on-secondary-fixed: '#001a42'
  on-secondary-fixed-variant: '#004395'
  tertiary-fixed: '#ffdbca'
  tertiary-fixed-dim: '#ffb690'
  on-tertiary-fixed: '#341100'
  on-tertiary-fixed-variant: '#783200'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: 0em
  headline-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  code-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0em
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0em
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 14px
    letterSpacing: 0.02em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  grid-unit: 8px
  space-2xs: 2px
  space-xs: 4px
  space-sm: 8px
  space-md: 12px
  space-lg: 16px
  space-xl: 24px
  space-2xl: 32px
  gutter-console: 8px
  panel-padding-compact: 12px
  panel-padding-standard: 16px
---

## Brand & Style

This design system delivers a high-density, mission-critical operational interface for disaster response coordinators and orbital telemetry specialists. The visual narrative is rooted in tactical command centers: authoritative, hyper-focused, and unyielding under pressure. Every visual decision prioritizes zero-latency data recognition, rapid situational assessment, and absolute operational clarity during high-stress crises.

The aesthetic fuses modern aerospace instrumentation with utilitarian software architecture. It relies on deep slate grounds, structural borders, razor-sharp monospaced data matrices, and high-visibility status illumination. Cognitive fatigue is mitigated by suppressing decorative ornamentation; decorative glows and soft gradients are strictly limited to active sensor traces, live target acquisitions, and escalating threat vectors.

## Colors

The color system operates on an absolute dark-mode baseline engineered for low-light multi-monitor workstations. Surfaces are structured in layered tonal slates to enforce spatial hierarchy without heavy drop shadows.

### Core Canvas & Surfaces
- **App Background:** `#0F172A` (Slate-900) — Deep operational void; anchor for global screen estate.
- **Surface / Panel Tier 1:** `#1E293B` (Slate-800) — Modular cards, telemetry docking zones, sidebars.
- **Surface / Panel Tier 2 & Framing:** `#334155` (Slate-700) — Active component containers, dividers, structural outlines.
- **Surface Hover / Highlight:** `#475569` (Slate-600) — Transient states, table row hover, selected card boundaries.

### Typography & Readout
- **Text Primary:** `#E2E8F0` (Slate-200) — Primary data values, critical alerts, high-priority labels.
- **Text Muted:** `#94A3B8` (Slate-400) — Secondary telemetry descriptors, inactive navigation, column headers.

### Threat, Sensor & Telemetry Tokens
- **Brand / Tactical Accent:** `#06B6D4` (Cyan-500) — Radar sweeps, targeted coordinate reticles, active tab bars.
- **Optical / Sensor Info:** `#3B82F6` (Blue-500) — Satellite telemetry downlinks, optical feeds, weather overlays.
- **Safe / Nominal:** `#22C55E` (Emerald-500) — Healthy uplink, stabilized sector, verified evacuation path.
- **Advisory / Warning:** `#F59E0B` (Amber-500) — Sensor drift, approaching front, staging zone caution.
- **Elevated Severity:** `#F97316` (Orange-500) — Immediate risk, structural integrity alert, trajectory breach.
- **Critical / Danger:** `#EF4444` (Red-500) — Disaster event epicenter, severed communication, mandatory egress.

## Typography

Typography is split into dual operational lanes: structural human interface copy (Inter) and machine telemetry readouts (JetBrains Mono).

- **Primary UI (Inter):** Applied across global shell navigation, section headers, procedural modals, and general status summaries. Strict geometric clarity guarantees immediate scanning under high information density.
- **Telemetry & Metrics (JetBrains Mono):** Mandatory for UTC timestamps, geo-coordinates (Lat/Long/Alt), raw sensor streams, orbital inclinations, and gauge readouts. Tabular figures prevent layout shifting during real-time data streaming.
- **Label Caps:** All panel micro-headers, metric units, and status badges render in uppercase monospaced text with widened letter-spacing (`0.08em`) to guarantee distinct legibility at small sizes.

## Layout & Spacing

The design system uses a strict 8px grid system optimized for dense multi-pane tactical arrangements. Space is utilized completely; large decorative dead-zones are strictly prohibited.

- **Layout Model:** A fluid CSS Grid / Flexbox layout structure with 8px gutters (`gutter-console`). The standard configuration anchors a central orbital/GIS viewport flanked by collapsible contextual telemetry docks (left: incident tree and active orbits; right: sensor telemetry, hazard assessment, and log streams).
- **Responsive Handling:**
  - **Desktop (1440px+):** Multi-dock arrangement. Dual 320px–380px fixed side-panels with a central fluid spatial canvas.
  - **Tablet / Lower Console (1024px – 1439px):** Secondary docks shift into overlay slide-ins or tabbed side-rails; viewport scales fluidly.
  - **Compact Displays (< 1023px):** Single-column stacked mode. Docks switch to full-screen modular drawers accessible via tactical toolbar keys.

## Elevation & Depth

Visual hierarchy is maintained through crisp structural borders, surface stepping, and precise light emission rather than diffuse environmental drop shadows.

- **Surface Stepping:** Background (`#0F172A`) supports Panel Surfaces (`#1E293B`), which in turn support Interactive Fields/Nested Pods (`#0F172A` inset or `#334155` elevated).
- **Crisp Structural Borders:** Panels and structural blocks are bounded by a uniform 1px solid border (`#334155`). Inactive boundaries remain subtle; focused or alerted panels transition their border color directly to `#06B6D4` (accent) or `#EF4444` (danger).
- **Glow & Status Lighting:** Environmental drop shadows are disabled across standard cards. In their place, tactical status components utilize tight, high-intensity perimeter glows to signal urgency (e.g., `box-shadow: 0 0 12px rgba(239, 68, 68, 0.45)` on critical alert modules and live ping indicators).

## Shapes

The geometric architecture balances modern usability with utilitarian modularity. Standard corner treatments adhere to a controlled hierarchy:

- **Panels & Dashboard Modules:** Uniform 8px radius (`0.5rem`). This bounds larger spatial regions while maintaining a structural, chassis-like container silhouette.
- **Badges, Status Chips, & Inputs:** Exact 4px radius (`0.25rem`). Gives compact components a sharp, calculated aesthetic that prevents wasted bounding box real estate.
- **Buttons & Tactical Controls:** 4px radius (`0.25rem`), preserving a disciplined instrumental switch feel.
- **Radar Reticles & Status Dots:** Fully circular (`9999px`) to maintain geometric clarity against cartographic coordinate grids.

## Components

### Buttons & Tactical Triggers
- **Primary:** Background `#06B6D4`, text `#0F172A`, font `Inter` semibold (13px), 4px border radius. Hover: `#22D3EE`. Focus: 2px ring `#06B6D4` with 2px offset.
- **Secondary / Surface:** Background `#1E293B`, 1px solid `#334155`, text `#E2E8F0`. Hover: `#334155`, border `#475569`.
- **Destructive / Emergency Action:** Background `#EF4444`, text `#FFFFFF`. Hover: `#DC2626`. Pulsing border outline for confirmed emergency triggers.
- **Size Profiles:** Dense vertical padding (`4px 10px` for micro tools, `8px 14px` for primary control switches).

### Mission Control Status Badges & Chips
- Monospaced typography (`JetBrains Mono`, 10px uppercase, `0.08em` tracking).
- 4px border-radius, `2px 6px` padding.
- Semi-transparent background (12% opacity) bounded by a solid 1px border matching the respective status token:
  - *Nominal:* Background `rgba(34, 197, 94, 0.12)`, Border `#22C55E`, Text `#22C55E`.
  - *Advisory:* Background `rgba(245, 158, 11, 0.12)`, Border `#F59E0B`, Text `#F59E0B`.
  - *Elevated:* Background `rgba(249, 115, 22, 0.12)`, Border `#F97316`, Text `#F97316`.
  - *Critical:* Background `rgba(239, 68, 68, 0.16)`, Border `#EF4444`, Text `#EF4444`, with a 1Hz keyframe beacon flash on live incident counters.

### Inputs & Telemetry Filters
- Background `#0F172A`, 1px border `#334155`, text `#E2E8F0`, placeholder `#94A3B8`.
- Height: 32px for maximum information density.
- Focus state: Border transitions to `#06B6D4` with an inner hairline highlight. No blurred outer rings.

### Modular Panels & Cards
- Background `#1E293B`, 1px solid border `#334155`, 8px corner radius.
- Header bar: Integrated 36px high bar with bottom 1px divider (`#334155`), housing the panel title in uppercase label-caps, telemetry mode indicator, and action icons.

### Telemetry Lists & Data Grids
- Alternating subtle row shading (`#1E293B` to `#182234`).
- Border-bottom 1px solid `#334155`.
- Data columns featuring coordinates, timestamps, or values strictly format in `JetBrains Mono` right-aligned with fixed tabular numbering.

### Checkboxes & Segmented Radios
- Box size: 14px x 14px with 2px radius. Border 1px solid `#475569`, background `#0F172A`.
- Selected state: Background `#06B6D4`, border `#06B6D4`, displaying a high-contrast `#0F172A` micro checkmark icon.
- Segmented Radios: Enclosed chassis container (`#0F172A`) with active segment sliding highlight in `#334155` and accent indicator bar.