# SIREN — Build Roadmap

**Companion to:** `docs/spec/PRD.md` (v4.3) · **Window:** 36-hour hackathon + pre-event prep
**Principle:** A complete evidence→review→dispatch loop with a rule-based change mask beats a sophisticated model that doesn't finish.

> **Build status:** Phases 0–6 complete. DoD chain verified end-to-end (104/104 tests passing). Phase 7 (rehearsal) pending.

---

## 0. Critical Path

```text
Data on disk ──► Change mask ──► Corridor (D8+OSM) ──► Exposure list ──► Scores ──► Review UI ──► Dispatch ──► Audit ──► Demo script
   (Phase 0)      (Phase 2)       (Phase 3)        (Phase 3)        (Phase 3)    (Phase 4)     (Phase 5)    (Phase 5)   (Phase 6)
```

Everything else — trained models, live APIs, polish — hangs off this spine. If any spine node is at risk, cut from the nearest non-spine work.

---

## Phase 0 — Pre-Event Prep (before the clock starts)

Do this at home. Every hour saved here is an hour of judging-visible work later.

- [x] **Lock the basin.** Dudh Koshi/Imja, Nepal — OSM data verified, Sentinel-1 pair downloaded.
- [x] **Download all data to disk:**
  - [x] Sentinel-1 GRD triplet: 2026-07-23 + 2026-08-04 + 2026-08-12 (Copernicus CDSE)
  - [x] Sentinel-2 L2A: 2025-11-22, tile T45RVL (clear-sky baseline)
  - [x] SRTM 1-arc-second DEM clip (srtm_30m.tif, 1188×1260)
  - [x] OSM extract: 1100 features — settlements, bridges, wells, clinics, rivers
  - [x] Weather series: `data/assets/weather_series.json` (obs-001 + obs-002 + obs-003)
- [x] **Verify the data opens:** all files load in rasterio/geopandas.
- [x] **Environment:** Python 3.14 venv with rasterio, geopandas, shapely, numpy, pysheds, fastapi. Node + Vite React.
- [x] **Repo scaffold:** `backend/`, `frontend/`, `data/{raw,processed,assets}/`, `docs/`.

**Exit criteria:** every dataset loads locally; D8 produces a flow-accumulation raster in a 30-second smoke test.

---

## Phase 1 — Foundation (Hours 0–4) ✅

| # | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | Define SQLite schema: `observations`, `runs`, `scores`, `reviews`, `dispatches`, `audit_log` | B | ✅ Schema in `db/schema.sql`; tables create cleanly |
| 1.2 | Ingest script: load baseline scene + assets into `data/processed/`, register in DB | A | ✅ Baseline mask generated; OSM extract loaded |
| 1.3 | Frontend shell: map view (MapLibre) + basin polygon + baseline raster overlay | C | ✅ MapView renders basin + layers |
| 1.4 | API skeleton: `GET /basin`, `GET /observations`, `POST /runs` | B | ✅ All endpoints return real data |

**Checkpoint (hour 4):** map shows the basin. If not, simplify — hardcode the basin GeoJSON, skip the DB for rasters (file paths only).

---

## Phase 2 — Change Detection Core (Hours 4–10) ✅

| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | Quality gate: cloud fraction (optical), alignment check, confidence multiplier → JSON verdict (PRD §9.1) | A | ✅ `preprocess/quality.py` — 11 tests |
| 2.2 | Optical path: NDWI differencing → water-change mask | A | ✅ `detect/ndwi.py` — baseline mask generated (11.88 km²) |
| 2.3 | SAR path: backscatter (VV/VH) ratio thresholding → water/debris mask | A | ✅ `detect/sar.py` — multi-look + slope masking |
| 2.4 | Weather-adaptive router: cloud ≥ 20% → SAR primary | B | ✅ `detect/router.py` — 33 lines |
| 2.5 | Change stats: area, % expansion per observation → DB | B | ✅ Pipeline writes `change_stats_json` |
| 2.6 | Timeline UI: observation list + change overlay toggle | C | ✅ TimelineView with card scrubbing + router strip |

**Checkpoint (hour 10):** the +8% and +28% expansion numbers from the demo script are computable from real masks. **Fallback:** if SAR preprocessing (calibration, orbit correction) stalls, use precomputed water masks for observations 2–4 and keep the pipeline code path for the live story.

---

## Phase 3 — Corridor, Exposure & Scores (Hours 10–16) ✅

| # | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | Combined corridor: D8 flow accumulation (reachability validation) + OSM river floodplain buffer (100–150 m) | A | ✅ `geo/corridor.py` — 295 lines, D8 + OSM buffering |
| 3.2 | Tolerance-buffer intersections: bridges ±75 m, roads ±50 m, settlements/wells ±100 m (PRD §6.4) | A | ✅ Exposures generated with correct buffers |
| 3.3 | Hazard score H (5-factor weighted, PRD §9.5) + confidence | B | ✅ `risk/fusion.py` — H + reasons in DB and API |
| 3.4 | Exposure priority E | B | ✅ E = H × Population Vulnerability × Infra Weight |
| 3.5 | Map: corridor + buffered asset overlays with severity styling | C | ✅ MapView renders corridor + asset markers |

**Checkpoint (hour 16):** the demo's "2 villages, 1 bridge, 3 wells" appears from real geometry, not hardcoded. **Primary method (validated):** combined D8 + OSM river buffering — D8 confirms the gravity gradient, OSM rivers capture the real surveyed riverbed through inhabited valleys (a raw D8 path can miss settlements in steep terrain). **Fallback:** if OSM rivers are absent, buffer the D8 path directly.

---

## Phase 4 — Disease Layer + Review Console (Hours 16–22) ✅

| # | Task | Owner | Done when |
|---|---|---|---|
| 4.1 | D_risk index: inundated water points × population density × temperature (PRD §9.5) | B | ✅ `risk/fusion.py::disease_risk()` |
| 4.2 | Disease Prevention Action Sheet generator (boil-water advisory, chlorine dispatch, per zone) | B | ✅ ReviewView renders per-well actions |
| 4.3 | Review panel UI: before/after swipe, scores + evidence reasons, asset list, action sheet, Confirm/Reject/Postpone buttons | C | ✅ ReviewView with two-step confirm + decision lock |
| 4.4 | Policy engine: informational / watch / elevated / critical-review thresholds | B | ✅ `risk/fusion.py::classify_severity()` — obs-002→critical, obs-001→watch |

**Checkpoint (hour 22):** the review card tells the full story without a presenter narrating.

---

## Phase 5 — Decision Loop: Dispatch + Audit (Hours 22–28) ✅

| # | Task | Owner | Done when |
|---|---|---|---|
| 5.1 | Confirm/Reject/Postpone workflow writes reviewer + timestamp + decision | B | ✅ `repo.create_review()` — 10 API tests |
| 5.2 | <250-byte compressed payload serializer (PRD §10.4) + size validator | B | ✅ `alerting/codec.py` — 12 tests, 118 bytes actual |
| 5.3 | Simulated dispatch: geofenced recipient groups, SMS/push/LoRa delivery table | B/C | ✅ AuditView channel simulator (SMS/LoRa/Satellite) |
| 5.4 | Append-only audit log: run, model version, inputs, decision, dispatch | B | ✅ `audit/writer.py` — 11 tests, triggers enforce immutability |
| 5.5 | Audit & dispatch panel UI | C | ✅ AuditView with byte meter + transmission preview |

**Checkpoint (hour 28):** clicking Confirm produces a visible dispatch + audit entry. **This is the spine's last node — nothing else matters if this breaks.**

---

## Phase 6 — Demo Wiring (Hours 28–32) ✅

- [x] Wire the 3-observation sequence end-to-end via **Run Monitoring** (sequential processing with visible progress)
- [x] Evidence explanation panel: ≥3 evidence factors on the elevated/critical alert (PRD §17.2) — obs-002 has 8 reasons, obs-003 has 8 reasons
- [x] **Stretch-goal gate:** core loop fully working → Search & Rescue Priority Layer ✅ BUILT (`risk/sar_priority.py` + `GET /runs/{id}/sar-priority`, 9 tests)
- [x] **ML evidence layer:** ✅ BUILT as optional evidence (`ml/` + `GET /runs/{id}/ml-evidence`, 13 tests, deterministic fallback when torch unavailable — ADR-002 addendum)
- [x] **SHA-256 audit hash chain:** ✅ BUILT (`audit/hash_chain.py`, tamper-evident lineage)
- [x] **Map asset endpoints:** ✅ BUILT (`api/map_assets.py` — DEM hillshade, SAR backscatter, baseline optical crops)
- [x] **Tailwind operational console:** ✅ BUILT (Ops Dark / Professional Light / Satellite themes, `frontend/src/theme/`)
- [x] **Docker deployment:** ✅ BUILT (`Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`, `start.sh`)
- [x] Polish pass: layer toggles, severity colors, empty states — all four views have empty/loading/error states

---

## Phase 7 — Rehearsal & Hardening (Hours 32–36)

- [ ] **Full offline rehearsal:** airplane mode on, run the entire demo script (PRD §16) twice
- [ ] **Backup video:** screen-record the complete demo; save locally + USB + cloud
- [x] Known-limitations doc (1 page): what's simulated, what's deterministic, latency realities — `docs/reference/KNOWN_LIMITATIONS.md`
- [ ] Pitch pass: 60-second narrative (overview doc §5), closing line, Q&A prep on scope questions ("Why no Area i?" → PRD §2 answer)
- [ ] Freeze: tag the demo commit. No new features after this point.

---

## Post-Build Enhancements (after Phase 6, before Phase 7)

The following enhancements were added after the core build was complete and the DoD chain was verified. None of these change the spine — they enhance the demo experience and communication of the two-tier alert routing concept.

### Alert Routing & Live Notifications

- [x] **Auto-SOS on CONFIRM:** Clicking CONFIRM in ReviewView fires a real ntfy.sh push notification automatically. Shared utility in `frontend/src/utils/ntfy.ts`. Toast confirms "Decision confirmed — SOS sent to phone". Does not violate Hard Rule #3 — human made the decision.
- [x] **ntfy.sh live phone alerts:** SMS channel sends real push notifications when online (gated by `navigator.onLine`). Topic: `siren-emergency-alert`. Urgent priority for sound + vibration.
- [x] **Secondary SEND TO PHONE:** AuditView retains a manual ntfy.sh send button for judges who want to see the FSM animation.
- [x] **First Responder Advisory row:** AuditView shows a pre-confirmation advisory row (amber border, "simulated" hash) when severity is elevated/critical and no decision yet. Communicates the two-tier routing concept. Disappears after confirmation.
- [x] **Escalation policy badge:** ReviewView header shows a static badge: "Advisory auto-routed to First Responders. Public broadcast held for Human Gate confirmation." Informational only — no auto-escalation dispatch.
- [x] **Early warning banner:** SimpleTriage mode shows "★ Early warning 12 days — trend flagged at obs-01 before critical threshold at obs-03".

### ReviewView Enhancements

- [x] **Simple/Advanced mode toggle:** Simple (Triage) mode shows satellite-first triage card with heatmap, chlorine logistics formula, and SOS checklist. Advanced (Analyst) mode shows full evidence panel, gauges, and reasons.
- [x] **Evidence thumbnails object-cover:** Before/after rasters and heatmap use `object-cover` (no letterboxing voids).

### AuditView Enhancements

- [x] **Real SHA-256 mock hashes:** Computed using the backend formula (`SHA256(prev_hash + timestamp + payload)`), not placeholders.
- [x] **Ledger JSON export:** Export the full audit ledger as a machine-readable JSON file.
- [x] **SitRep TXT export:** Export a field situation report as plain text.
- [x] **Web Crypto verification modal:** In-browser SHA-256 chain verification using `crypto.subtle.digest`. Shows "ALL 3 BLOCKS CRYPTOGRAPHICALLY LINKED" + "0 TAMPERING DETECTED".
- [x] **RF telemetry specs:** LoRa (868.1 MHz ISM, SF9, 125 kHz, 222 bytes max) and Iridium SBD (1621 MHz L-Band, 340 bytes/SBD).

### UI Polish Pass

- [x] **MapView:** Circular asset markers (border-radius:50%) + circular legend dots. Initial camera uses `jumpTo` (centered on Imja Lake) instead of `fitBounds`. Solid 3px corridor line (was thin dashed). Swipe compare uses absolute-positioned object-cover.
- [x] **TimelineView:** Single legend in chart header (removed duplicate SVG labels). Axis font sizes bumped (Y 11px, X 12px). Thumbnails use `object-cover`.
- [x] **Projector-ready typography:** Centralized type tokens in `tailwind.config.js` + `index.css`. Nav height 54px, banner 46px, label-caps 14px/600 weight.
- [x] **Nav tabs font-semibold:** All nav tabs use `font-semibold` baseline (not just active tab) for projector legibility.
- [x] **Removed tactical-bezel/tactical-reg from ReviewView header:** clip-path was clipping dropdowns; ::after pseudo-element was rendering floating ghost text.

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
| 16 | Does the corridor hit the demo assets? | Buffer the D8 path directly (no OSM rivers) |
| 22 | Does the review card render complete evidence? | Cut disease action sheet to a static template |
| 28 | Does Confirm → dispatch → audit work? | **All hands on this. Drop everything else.** |
| 32 | Is the full demo rehearsed once? | Cut the stretch goal; record backup video now |

---

## Definition of Done (maps to PRD §17.2) ✅

The MVP is done when, offline, in one click-chain: baseline loads → 3 observations process → elevated/critical card appears with ≥3 evidence factors → Confirm produces a ≤250-byte simulated dispatch → audit log reconstructs the full lineage with SHA-256 hash chain. Everything else is negotiable; that chain is not.

**Verified end-to-end:**
- ✅ Baseline loads (GET /basin returns Dudh Koshi/Imja with real AOI polygon)
- ✅ 3 observations process (POST /runs/process-all — obs-001: watch, obs-002: critical, obs-003: critical)
- ✅ Elevated/critical card with ≥3 evidence reasons (obs-002: 8 reasons, obs-003: 8 reasons)
- ✅ Human confirm (POST /runs/{id}/review with decision=confirm)
- ✅ ≤250-byte dispatch (118 bytes actual — POST /runs/{id}/dispatch)
- ✅ Audit lineage reconstructable with SHA-256 hash chain (GET /audit?run_id=...)
- ✅ 104/104 tests passing (101 active + 3 torch-gated)
