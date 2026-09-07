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

---

## Live Service Transition Roadmap (Post-Hackathon)

**Companion to:** ADR-006, ADR-007, ADR-008, ADR-009 · **Applies to:** transitioning SIREN from an offline demo to a continuously running hosted service.

**Principle:** the frozen deterministic pipeline is not changed. What changes is what feeds it and how often it is triggered. Phases build on each other — do not skip a go/no-go gate.

> **Critical blocker that applies to all phases:** `run_pipeline()` currently accepts only the three hardcoded demo observation IDs (`obs-001`, `obs-002`, `obs-003`). A live acquisition service can discover and download real satellite products, but the pipeline will reject them until the observation-acceptance interface is extended under an approved scope decision. Phase 4 is explicitly blocked until this is resolved. Phases 1–3 can proceed independently.

---

### Live Phase 0 — Frozen release and acceptance boundary

**Goal:** establish exactly what is frozen and what can change.

| Task | Done when |
|---|---|
| Pin the exact engine container image and processing version | A tagged, reproducible image exists; its digest is committed |
| Document every code/spec discrepancy | KNOWN_LIMITATIONS.md updated; all conflicts between PRD and implementation recorded |
| Verify the observation-acceptance interface | `run_pipeline("live-new-scene", repo)` → `ValueError` reproduced and documented; blocker formally recorded |
| Confirm storage/API adapter boundary | Written agreement on which interfaces can change (storage adapter, API layer) vs. which are frozen (pipeline internals) |
| Confirm ADR-002 compliance | ✅ Audit completed 2026-09-07: fusion weights diverge from PRD §9.5 (0.20 ML term), trend can be replaced by ConvLSTM, "SegFormer" is not SegFormer, ChangeFormer unimplemented — findings in `docs/reference/DL_MODEL_AUDIT.md`; ADR-010 proposed; accept/reject ADR-010 as part of the Phase 0 GO |

**GO criterion:** a written, agreed list of what is frozen and what can be changed exists before any live-service code is written.

**NO-GO action:** do not build Phase 1 until the boundary is agreed. Building acquisition infrastructure for a pipeline that cannot accept its output wastes the effort.

**Rollback:** none needed; this is a discovery phase only.

---

### Live Phase 1 — Reliable acquisition-only service

**Goal:** a service that discovers, downloads, verifies, and registers new satellite and weather products with durable job state — without triggering any pipeline run.

| Task | Done when |
|---|---|
| Fix `cdse.py`: current STAC endpoint, pagination, asset-role selection, atomic streaming download | Repeated discovery+download creates one canonical, verified local file; no partial files at destination |
| Add `acquisition_jobs` table (ADR-008 schema) | DB migration tested; `UNIQUE(source, provider_product_id)` prevents duplicates on scheduler retry |
| Wrap each ingest script in job-ledger logic | Every download attempt is recorded; failures are visible; exit 0 is never returned for a failed download |
| Fix `overpass.py`: add `waterway=river/stream`, emit flat properties, do not overwrite on empty response | Updated extract is compatible with the corridor module; an empty Overpass response is quarantined |
| Fix `openmeteo.py`: correct the backward date bug, make date window dynamic | Seven-day antecedent rainfall is computed correctly; missing precipitation is `null`, not `0` |
| Fix `srtm.py`: update to current Earthdata Cloud access URLs | Download completes without redirect failure |
| Add a simple scheduler (EventBridge or systemd timer) | Each source is polled at the recommended interval; missed polls are logged |
| Add credential management | Provider credentials stored in AWS Secrets Manager; never in source or environment variables committed to git |
| Add ingestion-health alerting | Missed heartbeats, persistent 401/403, and dead-letter jobs produce operator notifications on a separate channel from hazard alerts |

**GO tests:**
- Repeated scheduler triggers for the same satellite acquisition create exactly one `acquisition_jobs` row.
- A download that is interrupted mid-file never marks the job `verified` or `ready`.
- A 401 response triggers one token refresh; a second 401 marks the job `failed` and sends an operator alert.
- An empty Overpass response does not overwrite the last approved OSM extract.
- A missing IMERG granule is recorded as `null` rainfall, not `0`.

**NO-GO trigger:** ingestion success is still inferred from process exit code or file existence.

**Rollback:** disable the scheduler; the demo continues to work from prepared data unchanged.

---

### Live Phase 2 — Basin and input qualification

**Goal:** verify that the live acquisition pipeline produces inputs that the frozen corridor and detector implementations can use correctly, and that the Imja basin has adequate coverage for the intended monitoring purpose.

| Task | Done when |
|---|---|
| Expand acquisition inventory | At least two full orbit-12 and orbit-121 SAR acquisitions verified, downloaded, and stored |
| Validate SAR footprint covers Imja Lake | Actual pixel coverage at 86.925°E verified for each track; orbit-85 excluded from Imja assessments |
| Validate OSM river geometry compatibility | Updated Overpass extract opens correctly in the corridor module; river segment count is within expected range |
| Validate IMERG rainfall window | Seven-day window computed from real IMERG granules matches open reference for at least one historical date |
| Independent corridor evaluation | Corridor run on at least one real non-demo SAR scene; exposed assets visually checked against known valley geography |
| OSM population/freshness assessment | Critical assets reviewed against an authority source; population gaps documented; exposure reports labeled with data-source confidence |
| DEM qualification | SRTM version pinned; checksum committed; replacement policy defined |
| ML retraining per ADR-010 Stage 1 | Single-date SAR water segmenter trained on Sen1Floods11 with the **official event-level splits**; held-out-event evaluation passes agreed thresholds before any ML output is displayed as evidence |
| South Lhonak event-validation dataset | Pre/post Oct-2023 S1+S2 over South Lhonak lake and the Teesta corridor acquired and verified; covering orbits identified; retrospective "would SIREN have flagged it" run documented (`docs/reference/PRODUCTION_ML_PLAN.md` §3C/§4) |
| Dataset licensing review | Sen1Floods11 terms / WorldFloods (**CC non-commercial**) / SSL4EO-S12 / SegFormer & ChangeFormer code licenses reviewed against the deployment model |

**GO criteria:**
- At least two real lake-covering SAR acquisitions available and verified.
- Corridor produces plausible exposure output on a scene not from the demo pair.
- OSM freshness and population gaps are quantified and accepted by an operational partner.
- Scientific suitability is accepted independently of infrastructure uptime.

**NO-GO trigger:** required lake coverage is unavailable on the correct orbits; or corridor performance depends on the demonstration masks; or OSM population gaps are unacceptable to the operational partner.

**Rollback:** retain shadow observation mode; do not advance to Phase 3 with unqualified inputs.

---

### Live Phase 3 — Hosted persistence, security, and live UI

**Goal:** replace the single-machine demo stack with a multi-instance hosted service. Database, auth, API, and frontend changes. No pipeline changes.

| Task | Done when |
|---|---|
| Migrate schema to PostgreSQL / RDS (ADR-007) | Row counts and FK checks match the SQLite source; restore drill passes; multi-instance write test passes without ID collision |
| Implement storage adapter | Repository interface unchanged; adapter tested with both SQLite (demo) and PostgreSQL (hosted) backends |
| Add S3 object storage | Verified products stored at immutable keys; local materialization for execution; lifecycle rules for retention |
| Add OIDC authentication (ADR-009) | Reviewer identity from authenticated claims; ingestion-worker role cannot submit reviews |
| Add paginated API responses | `/observations`, `/runs`, `/audit` accept `?page=` and `?limit=`; no unpaginated list queries in production |
| Add health and freshness endpoints | `GET /health/ingestion` returns per-source last-success time, pending job count, and any dead-letter jobs |
| Update frontend: live timeline mode | Frontend polls for new observations without a simulation button; no automatic mock fallback on API error |
| Server-side delivery outbox (ADR-009) | `delivery_jobs` table; background worker; `"delivered"` means provider accepted; browser-side ntfy retained for demo only |
| Backup and restore | Automated daily RDS snapshots; quarterly restore drill; backup copies in a separate region |
| Docker / Compose / IaC updated | `docker-compose.yml` supports both demo (SQLite) and hosted (PostgreSQL + S3) profiles via environment variables |

**GO tests:**
- Two concurrent API instances write reviews without ID collision.
- A later `reject` decision suppresses an unsent dispatch from a prior `confirm`.
- API outage does not cause the frontend to display mock data as live.
- Browser restart restores actual review state from the server, not from localStorage cache.
- Restore drill reconstructs the database and referenced evidence correctly.
- Audit chain verification uses stored records; independent checkpoint passes.

**NO-GO trigger:** multi-instance writes produce ID collisions; or review suppression logic is incorrect; or restore drill fails.

**Rollback:** revert to single-instance SQLite deployment; demo unaffected.

---

### Live Phase 4 — Automatic scoring integration

**Prerequisite:** Live Phase 0 GO criterion must be met. **Currently blocked** by `run_pipeline()` accepting only demo observation IDs.

**Goal:** a verified, ready observation automatically triggers an authenticated pipeline run, producing a scored result for human review — without any browser action.

| Task | Done when |
|---|---|
| Resolve live-observation interface blocker | Scope decision made: either the observation-acceptance interface is extended, or an approved adapter is defined outside the frozen boundary |
| Implement input materialization | Processing worker copies pinned inputs from S3 to task-local ephemeral storage; never reads from a shared mutable path |
| Implement authenticated `POST /runs` trigger | Ready-input event submits an idempotent run request with a stable idempotency key |
| Implement processing worker | Worker acquires a lease; executes the frozen callable; publishes the complete result before releasing the lease |
| Wire complete result publication | Score, exposures, and audit entry are committed in a single transaction before the run is marked complete |
| Validate idempotency | Duplicate ready events produce exactly one run record |

**GO tests:**
- A new provider observation ID reaches the review queue without browser action.
- Duplicate submission of the same ready event yields one logical run and one review card.
- Worker crash is reconciled without duplicate public effects.
- Replaying the same input manifest produces identical scientific results.
- A missing required input cannot produce scenario-mask or seeded-default evidence.
- The ingestion-worker role cannot submit a review or trigger a dispatch.

**NO-GO trigger:** making the integration work requires changing frozen pipeline behavior or frozen scoring logic.

**Rollback:** disable automatic triggering; manual `POST /runs` from the API remains available.

---

### Live Phase 5 — Extended shadow operation

**Goal:** at least 30 days of unattended operation, supplemented by historical seasonal validation. No public alerts during this phase.

| Task | Done when |
|---|---|
| Monitor every expected acquisition | Each expected pass is accounted for: processed, rejected, unavailable, or failed with reason |
| Credential rotation drill | CDSE token refresh and Earthdata token rotation tested under real expiry conditions |
| Provider outage simulation | Backend unavailable for 24 hours; acquisition degrades gracefully; assessments labeled with data age |
| Disaster recovery drill | Full restore from backup to a clean environment; API and frontend reconnect correctly |
| Cost measurement | Actual AWS spend measured against the budget estimate; anomalies investigated |
| Scientific review | False-positive and false-exposure rate estimated from at least two independent no-change periods |
| Operator runbook | Runbook covers common failure scenarios; on-call rotation defined |

**GO criteria:**
- Every expected acquisition is accounted for (no unexplained gaps).
- No duplicate assessments or duplicate alert records.
- Storage and retry behavior is bounded (no unbounded growth observed).
- Scientific suitability accepted by the designated operational partner.
- Agreed freshness and operator-response targets are met in practice.

**NO-GO trigger:** unexplained data gaps; unbounded storage or retry growth; scientific performance below accepted thresholds.

---

### Live Phase 6 — Authority-supervised operational pilot

**Goal:** the first live hazard assessments with a real response partner. Human review and confirmation required before every dispatch. Alerts go to an approved recipient group, not the public ntfy topic.

| Task | Done when |
|---|---|
| Approved delivery provider | An authority-reviewed channel (SMS gateway, official push, or satellite messenger) is integrated and tested |
| Recipient groups defined | Groups, geofences, and escalation contacts are configured and approved by the operational partner |
| Review staffing | Coordinator schedule, review expiry, and escalation procedure are agreed and documented |
| Message template testing | Alert templates tested in relevant languages; LoRa/satellite byte constraints verified against Nepal radio regulations |
| First supervised dispatch | At least one full confirm→deliver cycle with receipt confirmation; reviewed by the operational partner |
| Public communication framing | All public-facing materials clearly label the service as supplementary decision support, not a replacement for official warning systems |

**GO criteria:**
- All dispatches trace to valid authenticated confirmations.
- Delivery receipts recorded; expired undelivered messages flagged.
- Ingestion-health notifications use a separate channel from hazard messages.
- Operational partner accepts the system and its limitations in writing.

**NO-GO trigger:** anyone depends on satellite polling as immediate GLOF detection; or simulated `"sent"` records are treated as confirmed delivery; or alert templates have not been validated in the target language and format.

---

### Live Service Go/No-Go Summary

| Phase | Key gate question | If NO |
|---|---|---|
| 0 | Is the frozen boundary agreed in writing? | Do not start Phase 1 |
| 1 | Are downloads atomic, verified, and idempotent? | Fix before scheduling |
| 2 | Do real acquisitions produce corridor-compatible inputs? | No automatic triggering until yes |
| 3 | Does multi-instance operation produce no ID collisions? | Fix before adding more instances |
| 4 | Is the live-observation blocker resolved? | Phase 4 stays blocked |
| 5 | Are data gaps, duplicates, and growth bounded after 30 days? | Extend shadow period |
| 6 | Has a real dispatch been confirmed, delivered, and receipted? | No public pilot |

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
