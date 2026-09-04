# SIREN — Build Roadmap

**Companion to:** `docs/PRD.md` (v4.0) · **Window:** 36-hour hackathon + pre-event prep
**Principle:** A complete evidence→review→dispatch loop with a rule-based change mask beats a sophisticated model that doesn't finish.

---

## 0. Critical Path

```text
Data on disk ──► Change mask ──► D8 corridor ──► Exposure list ──► Scores ──► Review UI ──► Dispatch ──► Audit ──► Demo script
   (Phase 0)      (Phase 2)       (Phase 3)        (Phase 3)        (Phase 3)    (Phase 4)     (Phase 5)    (Phase 5)   (Phase 6)
```

Everything else — trained models, live APIs, polish — hangs off this spine. If any spine node is at risk, cut from the nearest non-spine work.

---

## Phase 0 — Pre-Event Prep (before the clock starts)

Do this at home. Every hour saved here is an hour of judging-visible work later.

- [ ] **Lock the basin.** Decide Nepal (Dudh Koshi/Imja) vs. India (Chorabari or South Lhonak). Decision criteria: judge audience, OSM data completeness (check Overpass before committing), and whether you can get clean Sentinel-1 pairs covering a demo window.
- [ ] **Download all data to disk:**
  - [ ] Sentinel-1 GRD pair(s) — baseline + "current" (Copernicus CDSE)
  - [ ] Sentinel-2 L2A pair(s) — clear-sky baseline (for the NDWI path)
  - [ ] SRTM 1-arc-second DEM clip covering basin + full downstream corridor (not just the lake — the corridor extends far)
  - [ ] OSM extract: buildings, roads, bridges (`highway=bridge`), wells/water points, clinics, schools via Overpass → GeoPackage
  - [ ] GPM IMERG sample + Open-Meteo historical series for the demo dates
- [ ] **Verify the data opens:** load every file once in a notebook. Corrupt GeoTIFFs discovered at hour 20 are fatal; discovered now they're free.
- [ ] **Environment:** Python 3.11 venv/conda with rasterio, geopandas, shapely, numpy, whitebox (or pysheds), fastapi, uvicorn. Node + Vite React template. **Test WhiteboxTools D8 runs on the DEM now** — binary install issues are the #1 silent killer.
- [ ] **Repo scaffold:** `backend/`, `frontend/`, `data/{raw,processed,assets}/`, `docs/`. Commit the PRD.

**Exit criteria:** every dataset loads locally; D8 produces a flow-accumulation raster in a 30-second smoke test.

---

## Phase 1 — Foundation (Hours 0–4)

| # | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | Define SQLite schema: `observations`, `runs`, `scores`, `reviews`, `dispatches`, `audit_log` | B | Schema file committed; tables create cleanly |
| 1.2 | Ingest script: load baseline scene + assets into `data/processed/`, register in DB | A | Baseline renders as a map layer |
| 1.3 | Frontend shell: map view (MapLibre) + basin polygon + baseline raster overlay | C | Map shows basin + baseline in browser |
| 1.4 | API skeleton: `GET /basin`, `GET /observations`, `POST /runs` | B | Endpoints return real data |

**Checkpoint (hour 4):** map shows the basin. If not, simplify — hardcode the basin GeoJSON, skip the DB for rasters (file paths only).

---

## Phase 2 — Change Detection Core (Hours 4–10)

| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | Quality gate: cloud fraction (optical), alignment check, confidence multiplier → JSON verdict (PRD §9.1) | A | Gate output matches PRD contract |
| 2.2 | Optical path: NDWI differencing → water-change mask | A | Mask overlays on map, plausible water pixels |
| 2.3 | SAR path: backscatter (VV/VH) ratio thresholding → water/debris mask | A | Mask works on the cloudy "Observation 2" scene |
| 2.4 | Weather-adaptive router: cloud ≥ 20% → SAR primary | B | Router picks SAR for the monsoon scene automatically |
| 2.5 | Change stats: area, % expansion per observation → DB | B | `water_area_change_percent` populated |
| 2.6 | Timeline UI: observation list + change overlay toggle | C | Timeline renders all 4 demo observations |

**Checkpoint (hour 10):** the +8% and +28% expansion numbers from the demo script are computable from real masks. **Fallback:** if SAR preprocessing (calibration, orbit correction) stalls, use precomputed water masks for observations 2–4 and keep the pipeline code path for the live story.

---

## Phase 3 — Corridor, Exposure & Scores (Hours 10–16)

| # | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | D8 flow accumulation on DEM; extract downstream corridor from change polygon | A | Corridor polyline/polygon reaches downstream settlements |
| 3.2 | Tolerance-buffer intersections: bridges ±75 m, roads ±50 m, settlements/wells ±100 m (PRD §6.4) | A | Asset list matches demo script: 2 villages, 1 bridge, 3 wells |
| 3.3 | Hazard score H (5-factor weighted, PRD §9.5) + confidence | B | Score + per-factor reasons in DB and API |
| 3.4 | Exposure priority E | B | Ranked asset list in API |
| 3.5 | Map: corridor + buffered asset overlays with severity styling | C | Corridor and flagged assets visible on map |

**Checkpoint (hour 16):** the demo's "2 villages, 1 bridge, 3 wells" appears from real geometry, not hardcoded. **Fallback:** if D8 corridor misbehaves (spurious flow paths — common in karst/steep terrain), buffer the river network from OSM instead and intersect that.

---

## Phase 4 — Disease Layer + Review Console (Hours 16–22)

| # | Task | Owner | Done when |
|---|---|---|---|
| 4.1 | D_risk index: inundated water points × population density × temperature (PRD §9.5) | B | Wells flagged with risk values |
| 4.2 | Disease Prevention Action Sheet generator (boil-water advisory, chlorine dispatch, per zone) | B | Sheet renders per affected zone |
| 4.3 | Review panel UI: before/after swipe, scores + evidence reasons, asset list, action sheet, Confirm/Reject/Postpone buttons | C | Full card renders for the elevated observation |
| 4.4 | Policy engine: informational / watch / elevated / critical-review thresholds | B | Observation 2 → Elevated; Observation 1 → Advisory |

**Checkpoint (hour 22):** the review card tells the full story without a presenter narrating.

---

## Phase 5 — Decision Loop: Dispatch + Audit (Hours 22–28)

| # | Task | Owner | Done when |
|---|---|---|---|
| 5.1 | Confirm/Reject/Postpone workflow writes reviewer + timestamp + decision | B | Decision persists; rejected alerts don't dispatch |
| 5.2 | <250-byte compressed payload serializer (PRD §10.4) + size validator | B | Payload ≤ 250 bytes, round-trips |
| 5.3 | Simulated dispatch: geofenced recipient groups, SMS/push/LoRa delivery table | B/C | Dispatch log renders with statuses |
| 5.4 | Append-only audit log: run, model version, inputs, decision, dispatch | B | Full lineage reconstructable from DB |
| 5.5 | Audit & dispatch panel UI | C | Panel shows decision timeline + delivery log |

**Checkpoint (hour 28):** clicking Confirm produces a visible dispatch + audit entry. **This is the spine's last node — nothing else matters if this breaks.**

---

## Phase 6 — Demo Wiring (Hours 28–32)

- [ ] Wire the 4-observation sequence end-to-end via **Run Monitoring** (sequential processing with visible progress)
- [ ] Evidence explanation panel: ≥3 evidence factors on the elevated alert (PRD §17.2)
- [ ] **Stretch-goal gate:** core loop fully working? → build the Search & Rescue Priority Layer (population × access-loss ranking, ~3–4 hrs). Core loop shaky? → spend the hours on polish and rehearsal instead.
- [ ] Polish pass: layer toggles, severity colors, empty states

---

## Phase 7 — Rehearsal & Hardening (Hours 32–36)

- [ ] **Full offline rehearsal:** airplane mode on, run the entire demo script (PRD §16) twice
- [ ] **Backup video:** screen-record the complete demo; save locally + USB + cloud
- [ ] Known-limitations doc (1 page): what's simulated, what's deterministic, latency realities
- [ ] Pitch pass: 60-second narrative (overview doc §5), closing line, Q&A prep on scope questions ("Why no Area i?" → PRD §2 answer)
- [ ] Freeze: tag the demo commit. No new features after this point.

---

## Team Work Streams

| Stream | Owner | Owns phases | Spine nodes |
|---|---|---|---|
| **Geospatial** | A | 0 (data), 2 (masks), 3 (corridor, buffers) | Change mask, corridor |
| **Backend/Pipeline** | B | 1 (schema), 2 (router, stats), 3 (scores), 4 (disease), 5 (dispatch, audit) | Scores, dispatch, audit |
| **Frontend** | C | 1 (map), 2 (timeline), 3 (overlays), 4 (review card), 5 (panels) | Review UI |

**Solo variant:** run phases strictly in order 0→7; skip 4.3 polish until Phase 6; the stretch goal is off the table.

**Pairing rule:** when a stream finishes early, its owner pairs on the next spine node — never starts non-spine work (extra ML, extra views) while the spine is incomplete.

---

## Go/No-Go Decision Points

| Hour | Question | If NO |
|---|---|---|
| 4 | Does the map render the basin? | Hardcode basin GeoJSON; defer DB |
| 10 | Are change masks plausible on both scenes? | Precompute masks manually; keep pipeline code for the story |
| 16 | Does the corridor hit the demo assets? | Fall back to OSM river buffering |
| 22 | Does the review card render complete evidence? | Cut disease action sheet to a static template |
| 28 | Does Confirm → dispatch → audit work? | **All hands on this. Drop everything else.** |
| 32 | Is the full demo rehearsed once? | Cut the stretch goal; record backup video now |

---

## Definition of Done (maps to PRD §17.2)

The MVP is done when, offline, in one click-chain: baseline loads → 4 observations process → elevated card appears with ≥3 evidence factors → Confirm produces a ≤250-byte simulated dispatch → audit log reconstructs the full lineage. Everything else is negotiable; that chain is not.
