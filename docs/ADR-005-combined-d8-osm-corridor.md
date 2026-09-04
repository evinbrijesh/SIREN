# ADR-005 — Combined D8 + OSM River Buffering Corridor

**Status:** Accepted · **Date:** 2026-09-04 · **Applies to:** exposure corridor generation

## Context

The exposure corridor must identify which settlements, bridges, and wells are downstream of a change source (e.g., the Imja glacial lake). A pure D8 steepest-descent path from the change polygon was tested on the real SRTM DEM and **missed all settlements** — the nearest was 2.8 km away. At 30 m resolution in steep Himalayan terrain, DEMs suffer drainage-trenching artifacts: narrow gorges, lateral moraine walls, and interpolation errors divert a single-pixel D8 path over a ridge or into a dry side-gully instead of keeping it in the deeply carved valley bed.

## Decision

Use a **combined D8 + OSM river buffering** corridor:

1. **D8 reachability (physical validation):** trace the downstream flow path from the change source to confirm it drains into the expected sub-basin (Imja lake → Imja Khola / Dudh Koshi), not an adjacent drainage divide.
2. **OSM river selection:** select `waterway=river/stream` segments reachable by the D8 path (within a reachability radius). OSM captures the real, surveyed riverbed through inhabited valleys.
3. **Floodplain buffer:** buffer the reachable river segments by 100–150 m.
4. **Exposure intersection:** intersect the buffered corridor against OSM assets using the PRD §6.4 tolerance buffers (bridges ±75 m, roads ±50 m, settlements/wells ±100 m).

## Consequences

- **Positive:** the corridor reliably catches the demo assets — validated on real data: Benkar (75 m), Jorsale (69 m), the Hillary suspension bridges, and drinking wells along the Dudh Koshi. D8 still provides the physical gravity-gradient check.
- **Negative:** depends on OSM river completeness. If OSM rivers are absent, fall back to buffering the D8 path directly.
- **Deterministic:** same inputs → same corridor. No learned components (ADR-002).

## Rationale

A raw D8 path is hydrologically correct but geometrically unreliable at 30 m in steep terrain for *exposure* purposes. OSM rivers are the surveyed ground truth of where water actually flows through inhabited valleys. Combining them gives physical validation (D8) + operational accuracy (OSM). This is the Roadmap Phase 3 fallback, promoted to the primary method after validation on real data.