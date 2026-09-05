---
name: Emergency Operations Console
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
  surface-canvas: '#0F172A'
  surface-panel: '#1E293B'
  surface-recessed: '#0A0F1E'
  border-subtle: '#334155'
  text-primary: '#E2E8F0'
  text-dim: '#94A3B8'
  status-safe: '#22C55E'
  status-warn: '#F59E0B'
  status-elevated: '#F97316'
  status-danger: '#EF4444'
  status-info: '#3B82F6'
  overlay-backdrop: rgba(0, 0, 0, 0.5)
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
  headline-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 24px
  headline-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  body-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  caption:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 14px
  metric-display:
    fontFamily: JetBrains Mono
    fontSize: 24px
    fontWeight: '400'
    lineHeight: 28px
  code-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  space-2: 2px
  space-4: 4px
  space-6: 6px
  space-8: 8px
  space-12: 12px
  space-16: 16px
  space-24: 24px
  nav-height: 56px
  banner-height: 44px
  dock-left-width: 200px
  dock-right-width: 240px
  decision-bar-height: 52px
  footer-height: 28px
---

## Brand & Style

This design system is built for the high-stakes, crisis-management context of the Satellite-Informed Risk & Emergency Network (SIREN) monitoring Himalayan glacial basins. In critical civil protection environments, visual noise, ambiguity, and latency in comprehension carry real-world consequences. The guiding mandate is **clarity over spectacle**. 

The brand persona is calm, analytical, authoritative, and strictly utilitarian. It functions as an unyielding command console rather than a consumer dashboard. The visual style pairs disciplined technical minimalism with structural data density:
- Zero decorative gradients, lighting sweeps, ambient drop shadows, or frosted glass blurs.
- Crisp boundary definitions that separate operational workspaces without visual bleed.
- Restrained color application where semantic hues are conserved purely for urgent state signaling (Safe, Advisory, Elevated, Danger) and active telemetry paths.
- Reliance on pure Unicode and typographic glyphs in place of decorative icon kits to minimize visual parsing overhead.

## Colors

The system runs exclusively in a fixed dark mode calibrated for 24/7 low-light monitoring centers, minimizing eye strain across 12-hour operational shifts.

### Surface Tiers
- **Canvas (`#0F172A`)**: Base environment layer representing global space.
- **Panel Surface (`#1E293B`)**: Standard containment for observation docks, inspector panels, timeline controllers, and modal bodies.
- **Recessed Surface (`#0A0F1E`)**: Low-reflectance floor reserved for GIS satellite maps, raw JSON audit logs, code payload blocks, and high-contrast telemetry viewports.

### Functional Roles
- **Accent (`#06B6D4`)**: Used for the core interactive controls, active observation borders, timeline execution pointers, and vector basin AOI perimeters.
- **Structural Border (`#334155`)**: Strict 1px division for cards, dock perimeters, headers, and disabled status meter tracks.
- **Text Primary (`#E2E8F0`)**: Crisp, high-legibility off-white for data points, titles, and critical notices.
- **Text Dim (`#94A3B8`)**: Mid-tone slate for supporting metadata, field titles, table column headers, and structural units.

### Semantic Triage Scale
Color is reserved strictly for situational status and telemetry:
- **Safe (`#22C55E`)**: Confirmed normal status, intact infrastructure, positive capacity.
- **Warn / Advisory (`#F59E0B`)**: Advisory watches, flow routing lines, delayed or queued channels.
- **Elevated (`#F97316`)**: Escalation stages requiring review, alert banner callout triggers.
- **Danger / Critical (`#EF4444`)**: Inundated assets, breach alerts, rejected missions.
- **Info (`#3B82F6`)**: Satellite optical masks, secondary sensor readings, standard audit stamps.

## Typography

The type architecture strictly enforces a dual-engine rule:
1. **Interface Engine (`Inter`)**: Governs structural navigation, headers, evidence descriptions, field titles, table data, and action labels. Avoids all decorative, extended, or stylized weights; stays anchored between 400 (Regular) and 500 (Medium).
2. **Telemetry Engine (`JetBrains Mono`)**: Strictly applied to non-prose machine outputs: risk gauge scores, satellite sensor telemetry IDs, ISO 8601 timestamps, packet payloads, and file masks.

### Behavioral Rules
- Headers remain in clean sentence case. Do not apply wide tracking (`letter-spacing`) or all-caps styling to container headers.
- Badges and status chips use uppercase for rapid spatial scanning, but maintain standard tracking without artificial horizontal expansion.
- Numerical readouts on meters and gauges must always align to monospace metrics to avoid horizontal jitter during continuous live stream updates.

## Layout & Spacing

Layout geometry follows an exact 8px base coordinate system, with a 4px sub-grid for micro-alignments (progress meters, badge padding, compact chip labels).

### Spatial Constraints
- **Application Shell**:
  - Global Navigation Header: Fixed `56px` height.
  - Critical Alert Bar: Fixed `44px` height directly below navigation.
  - Global Telemetry Footer: Fixed `28px` height.
  - Bottom Sticky Decision Bar: Fixed `52px` height.
- **Three-Tier Operational Map Layout**:
  - Left dock: Fixed width `200px` for toggles, layer stacks, and optical switches.
  - Center workspace: Fluid-stretch viewport hosting the satellite canvas (`#0A0F1E`).
  - Right dock: Fixed width `240px` for telemetry readouts, risk dials, and asset inventories.
  - Gap spacing between docks and main canvas is fixed at `12px`.
- **Review / Verification Layout**: 
  - Asymmetric split: `45%` evidence/input viewport vs `55%` simulation outcome and authorization panel.
- **Paddings**:
  - Global panel interior padding: `16px`.
  - Dense payload and notification interior padding: `12px`.
  - Vertical list gaps (toggles, asset lists, parameter rows): `8px`.

## Elevation & Depth

This system avoids atmospheric lighting, fake dimensionality, and ambient drop shadows entirely. Elevation is communicated strictly through planar surface contrast and explicit borders.

### The Planar Hierarchy
1. **Recessed Canvas Tier (`#0A0F1E`)**: The ground plane. Represents raw geographic context, map projection buffers, and raw code output.
2. **Workspace Canvas (`#0F172A`)**: Base shell background for all standard operational views.
3. **Structured Surface Tier (`#1E293B`)**: All panels, cards, data docks, and toolbars exist on this tier, bounded by a 1px solid stroke (`#334155`).
4. **Active/Focused Tier**: Denoted without Z-axis elevation. Active observation cards and selected toolsets are indicated by an explicit 2px border in Accent (`#06B6D4`).
5. **Modal System Layer**: Floating validation and dispatch dialogs (max-width `440px`) sit on a flat backdrop of `rgba(0, 0, 0, 0.5)` with no shadow diffusion.

## Shapes

The design system employs a disciplined, soft-geometric shape language optimized for technical density:

- **Panels, Cards, and Modals (`8px`)**: Container-level structures (cards, docks, payload boxes, dialogs) carry a consistent `8px` corner radius. This provides clean visual separation from the viewport edges while keeping layouts compact.
- **Interactive & Telemetry Components (`4px`)**: Action buttons, badge indicators, meter tracks, risk gauge bars, and byte capacity indicators use a minimal `4px` radius.
- **Circular Indicators (Full Round / `50%`)**: Reserved strictly for geographic map pinpoints (12px), legend state markers (10px), and the brand identity mark (`◉`).

## Components

### Buttons & Interactive Controls
- **Primary CTA**: Solid Accent fill (`#06B6D4`), dark background text (`#0F172A`), font size 14px Medium, border radius `4px`.
- **Action / Decision Confirm ("✓ Confirm")**: Solid Safe fill (`#22C55E`), dark text (`#0F172A`), radius `4px`.
- **Action / Decision Reject ("✗ Reject")**: Solid Danger fill (`#EF4444`), white text (`#FFFFFF`), radius `4px`.
- **Action / Decision Postpone ("⏸ Postpone")**: Solid Warn fill (`#F59E0B`), dark text (`#0F172A`), radius `4px`.
- **Ghost Utility Buttons ("Fly to", "Preview", "Copy")**: Transparent background, 1px border (`#334155`), text in `#94A3B8` transitioning to `#E2E8F0` on hover.

### Status Badges
- **Architecture**: Always hollow/transparent background with a 1px solid stroke matching the status color. No filled pill backgrounds.
- **Sizing**: `2px` vertical padding, `8px` horizontal padding, `4px` corner radius.
- **Type**: 12px Regular Inter, uppercase text colored identically to the perimeter stroke:
  - Safe: Border `#22C55E`, Text `#22C55E`
  - Advisory/Warn: Border `#F59E0B`, Text `#F59E0B`
  - Elevated: Border `#F97316`, Text `#F97316`
  - Danger/Critical: Border `#EF4444`, Text `#EF4444`

### Status Meters & Risk Gauges
- **Track**: Fixed height `4px` or `6px`, solid recessed background (`#0A0F1E`) with a 1px border (`#334155`). Radius `4px`.
- **Fill**: Solid block fill mapped directly to semantic triage colors without internal gradients.

### Informational Callouts & Alert Banners
- **Alert Banner**: Fixed `44px` height, `#1E293B` background, with an asymmetric `3px` solid stroke along the left border in the active severity color (`#F97316` or `#EF4444`).
- **Confirmation/Guidance Callout**: `#1E293B` panel with a `3px` solid `#22C55E` left border.

### Telemetry Code / Payload Viewers
- Contained within recessed `#0A0F1E` surfaces, bounded by 1px `#334155` borders. Font set to 12px or 14px `JetBrains Mono` at `#94A3B8` for syntax and `#E2E8F0` for active values.