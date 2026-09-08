# SIREN — Known Limitations

**Purpose:** One-page reference for judges on what is simulated, what is deterministic, and where the demo diverges from a production system. Read alongside `docs/spec/PRD.md` §14 (Explicit Scope Boundaries) and §18 (Future Roadmap).

---

## What's Deterministic (Real Code, Real Data)

- **Change detection.** NDWI differencing on Sentinel-2 and VV/VH backscatter ratio thresholding on Sentinel-1 are real rule-based implementations running on actual downloaded scenes. No pre-baked masks — the pipeline computes them from rasters at runtime.
- **D8 flow corridor.** pysheds D8 flow accumulation runs on the real SRTM 1-arc-second DEM clip (1188×1260). Combined with OSM river buffering for the surveyed riverbed.
- **Tolerance-buffer intersections.** Bridges ±75 m, roads ±50 m, settlements/wells ±100 m — computed against the real OSM extract (1100 features) using shapely.
- **Risk fusion.** H, E, D_risk, and confidence scores use the fixed PRD §9.5 weights (0.30/0.25/0.20/0.15/0.10). Same inputs → identical outputs. No unseeded randomness.
- **Payload codec.** The ≤250-byte compressed JSON is a real encoder/decoder with round-trip tests. 118 bytes actual.
- **Audit log.** Append-only, enforced by SQLite triggers (no UPDATE/DELETE paths exist). Lineage is queryable by alert_id or run_id. SHA-256 hash chain (`prev_hash` + `event_hash`) makes the log tamper-evident.
- **Human gate.** No code path dispatches without a recorded `confirm` review. Enforced at the DB layer.
- **SAR priority ranking.** `risk/sar_priority.py` computes a priority score for exposed assets (PRD §15). Real code on real OSM exposure data — 9 tests.
- **ML evidence layer (optional, audited 2026-09-07).** `ml/` contains a Siamese U-Net (trained on synthetic bi-temporal Sen1Floods11 pairs), a changed-crop classifier internally named "SegFormer" (not the SegFormer architecture; trained on threshold-generated weak labels), and a ConvLSTM trend model (trained on synthetic mask progressions). ChangeFormer is named in the PRD but not implemented. The deterministic path remains the default, but the audit found the ML layer is not currently qualified for live hazard assessment — see `docs/reference/DL_MODEL_AUDIT.md` and ADR-010 (Proposed).

## What's Simulated (Not Real at Runtime)

- **Alert channels — LoRa and Satellite.** LoRa and Satellite dispatch are simulated state machines (QUEUED → TRANSMITTING → DELIVERED). No real radio modems or Iridium SBD transceivers are contacted. The AuditView channel simulator shows delivery states for demo purposes.
- **Alert channel — SMS (partially live).** SMS is the only live integration: clicking the SMS channel (or clicking CONFIRM in ReviewView) sends a real ntfy.sh push notification when online. When offline (air-gap mode), the network request is skipped and the dispatch is simulated. ntfy.sh is a free push notification service — install the app and subscribe to topic `siren-emergency-alert` to receive alerts on your phone.
- **First Responder Advisory.** The advisory row in AuditView is a simulated visual showing the two-tier routing concept. No real pre-confirmation notification is sent to hospitals, firefighters, or SAR teams. It communicates what *would* happen in a production system.
- **Escalation policy badge.** The badge in ReviewView is informational only. No auto-escalation dispatch fires without human confirmation (Hard Rule #3). It shows the policy, not an automated action.
- **Live satellite ingestion.** All scenes are pre-downloaded to `data/raw/`. The `ingest/` scripts can fetch from CDSE/Earthdata/Overpass, but the runtime demo makes zero network calls for pipeline data (ADR-004: offline-first). The only runtime network call is the ntfy.sh push on CONFIRM, gated by `navigator.onLine`.
- **Weather data.** The rainfall series is a prepared JSON file (`data/assets/weather_series.json`), not a live API call. Open-Meteo integration exists in `ingest/` but is not used at runtime.
- **Synchronous pipeline.** `POST /runs` processes the full detect→geo→risk chain synchronously in the request. No background task queue (Celery/RQ). This is intentional for demo simplicity — a production system would use async workers.
- **Single reviewer.** The demo hardcodes `coordinator-01` as the reviewer identity. No authentication or RBAC.

## Known Data Gaps

- **Sentinel-1 swath coverage.** The ascending-orbit S1 pair (relative orbit 85) covers only the western AOI; the Imja lake proper (86.925°E) is outside the swath. A descending-pass pair (2026-07-02 + 07-14) was downloaded from CDSE to cover Imja, but is not yet wired into the demo pipeline. All 3 demo observations use real calibrated Sentinel-1 VV/VH sigma0 dB from ESA SAFE archives. The SAR pipeline code is real and validated on the covered region.
- **Single basin.** Only Dudh Koshi / Imja is configured. Multi-basin support is a V2 roadmap item.
- **No ground-truth validation set.** Change masks are not validated against a held-out labeled flood dataset. The Sen1Floods11 benchmark is referenced in the PRD for future calibration.

## Latency Realities

- **Pipeline processing time:** ~2–4 seconds per observation on the demo hardware (Docker container, single core). A production system with GPU-accelerated inference and parallel tile processing would target <30 seconds.
- **Dispatch latency:** Simulated as instant. Real LoRa mesh delivery in Himalayan terrain is 30–120 seconds; satellite SBD is 1–5 minutes; SMS depends on tower availability (which is the failure case SIREN is designed for).
- **Review latency:** Depends entirely on the human coordinator. SIREN shows an escalation policy badge communicating the two-tier routing concept, but does not auto-escalate on timeout — that's a policy decision for the deploying authority. The ntfy.sh push on CONFIRM is a side-effect of the human decision, not an autonomous dispatch.

## What SIREN Does NOT Do (PRD §14)

- Predict the exact time of glacial-lake collapse
- Guarantee exact flood depth or flow path
- Diagnose specific diseases from satellite imagery
- Issue autonomous evacuation orders
- Guarantee delivery to every person in an area
- Replace government early-warning systems
- Identify individuals, locate trapped persons, or handle family reunification (V4 roadmap, requires authority partnership)

---

**Bottom line for judges:** The detection, corridor, exposure, scoring, and audit chain is real code on real data. The SMS channel is live via ntfy.sh (when online); LoRa and Satellite are simulated. The First Responder Advisory and escalation policy badge communicate the two-tier routing concept without violating the human gate. SIREN is a decision-support and resilience layer, not a replacement for emergency infrastructure. The system deploys via Docker Compose (`./start.sh`) for a one-command demo.

---

## Production Transition Gaps

The following gaps were identified during the live-service architecture audit (2026-09-07). They do not affect the hackathon demo but must be resolved before the service runs unattended. Each item maps to a phase in the Live Service Transition Roadmap (`docs/spec/BUILD_ROADMAP.md`).

### Pipeline observation acceptance (Phase 4 blocker) — RESOLVED 2026-09-07
`run_pipeline()` now accepts both demo observations (`obs-001/002/003`, using hardcoded config and scenario masks) and live observations registered in the database via `repo.register_observation()`. Live observations must have a `raster_uri` pointing to a pre-computed change mask. The frozen deterministic pipeline semantics are unchanged. The Live Phase 4 blocker is resolved.

### Ingest scripts exit 0 on failure (Phase 1) — RESOLVED 2026-09-07
All four ingest scripts now support a `--strict` flag that returns non-zero exit codes on any failure (for scheduler use). Without `--strict`, network failures still exit 0 (offline-safe per ADR-004). API errors (empty results, invalid JSON) now return exit 1 even without `--strict`.

### CDSE uses deprecated endpoint (Phase 1) — RESOLVED 2026-09-07
`cdse.py` now targets `stac.dataspace.copernicus.eu/v1/search` (the active endpoint). The legacy `catalogue.dataspace.copernicus.eu/stac/search` endpoint was deprecated on 2025-11-17. Pagination via `links[].rel == "next"` is now supported.

### CDSE downloads entire archive into memory (Phase 1) — RESOLVED 2026-09-07
Downloads now stream to disk in 64 KiB chunks via `_http_stream_to_file()`, avoiding loading ~1.7 GB S1 GRD archives into memory.

### Overpass query missing river geometry (Phase 1, ADR-005 blocker) — RESOLVED 2026-09-07
`overpass.py` now queries `waterway=river` and `waterway=stream` (both nodes and ways). Automated refresh will include river geometry compatible with the corridor module.

### Overpass output format incompatible with corridor module (Phase 1) — RESOLVED 2026-09-07
Properties are now emitted FLAT: OSM tags are merged into top-level properties (`waterway`, `highway`, `amenity`, etc.) so the corridor module can filter directly. The nested `tags` key is also preserved for full-fidelity access. Empty Overpass responses no longer overwrite existing extracts.

### `openmeteo.py` date window walks forward, not backward (Phase 1) — RESOLVED 2026-09-07
The 7-day antecedent rainfall window now walks BACKWARD from the observation date: `[obs_date - 6, obs_date]`. Missing precipitation is recorded as `null` (not `0.0`), and the number of missing days is tracked in `rainfall_7d_days_missing`. The script now accepts `--lat`, `--lon`, `--date`, and `--out` CLI arguments.

### No job ledger or idempotency constraints (Phase 1) — PARTIALLY RESOLVED 2026-09-07
An `acquisition_jobs` table (ADR-008 schema) has been added with `UNIQUE(source, provider_product_id)` for idempotency. Repository methods (`create_acquisition_job`, `update_acquisition_job`, `find_acquisition_job`, `list_acquisition_jobs`) are available. The count-based run ID collision issue remains open for Phase 3.

### SRTM uses outdated access URL (Phase 1) — RESOLVED 2026-09-07
`srtm.py` now uses the LP DAAC Earthdata Cloud endpoint (`lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MEASURES/SRTMGL1.003/`). Bearer token auth (`EARTHDATA_TOKEN`) is supported alongside Basic auth. Downloads stream to disk.

### Ascending Sentinel-1 swath misses Imja Lake (partially resolved)
The ascending-orbit S1 pair (relative orbit 85) does not cover Imja Lake at 86.925°E. A descending-pass pair (2026-07-02 + 07-14) was downloaded from CDSE to cover Imja, but is not yet wired into the demo pipeline. All 3 demo observations now use real calibrated Sentinel-1 VV/VH sigma0 dB from ESA SAFE archives (obs-003 downloaded 2026-09-08). For operational monitoring, orbit 12 (ascending) and orbit 121 (descending) are the covering tracks, each with a 12-day repeat.

### Population defaults are fabricated (Phase 2)
Unknown assets are inserted with `population = 1,240` (a hardcoded default). There is no measured or authority-verified population figure for most exposed settlements. The OSM extract has one `population` field across 5,691 features, and no `survey:date` or `check_date` tags. Exposure reports must label population figures with their source and confidence.

### SQLite not suitable for multi-instance hosted operation (Phase 3, ADR-001 addendum)
The current in-memory-default SQLite configuration, count-based IDs, and WAL limitations are incompatible with a multi-process, multi-instance hosted service. See ADR-001 addendum and ADR-007.

### Review suppression logic incorrect (Phase 3) — RESOLVED 2026-09-08
`_confirm_review_for_run()` now checks the LATEST review decision for the run, not just any historical confirm. A confirm followed by a reject/postpone correctly suppresses dispatch (Hard Rule 3). Covered by `test_dispatch_after_confirm_then_reject_returns_409`.

### ntfy.sh delivery is browser-side and unverified (Phase 3, ADR-009)
Live alert delivery is a browser-side HTTP call. `"status": "sent"` in the database does not mean the message was delivered. Closing the browser tab or a network error during the ntfy POST silently loses the alert. A server-side delivery outbox with receipts and retry is required for operational use.

### Frontend automatic mock fallback on any API error (Phase 3)
The frontend API client falls back to mock data on any HTTP error, including server errors and authentication failures. In a live service, this can display stale or simulated assessments without any indication that the backend is unavailable.

### Audit `_audit()` call does not commit (Phase 3) — RESOLVED 2026-09-08
`Repository._audit()` now commits after every insert. The pipeline's final audit entry (the "run" log) is durable regardless of whether the caller commits afterward. Covered by `test_repo_audit_commits_immediately`.

### Audit hash verification uses recomputed hashes (Phase 3) — RESOLVED 2026-09-08
`list_audit()` now returns the STORED `prev_hash`/`event_hash` from the DB, not recomputed values. A new `verify_hash_chain()` method recomputes from stored `detail_json` and compares against stored hashes to detect tampering. Covered by `test_list_audit_returns_stored_hashes` + `test_verify_hash_chain_detects_tampering`.

### No S3 or object-storage integration (Phase 3)
Raw SAFE archives (approximately 1.7 GB per SAR product) are stored on the local container filesystem. At 5–10 SAR products per month, local storage will exhaust typical ECS ephemeral limits within weeks. S3 with immutable keys and lifecycle rules is required for durable raster storage.

### DL layer (audited 2026-09-07 — see docs/reference/DL_MODEL_AUDIT.md)

- **ML train/inference input mismatch (Phase 4).** The Siamese model trains on SAR chips but the pipeline feeds it binary water masks as both inputs.
- **Crop classifier can remove rule-detected evidence (Phase 4).** The Stage-2 `filtered_mask` replaces the consensus mask and can delete entire rule-detected regions; the deterministic fallback heuristic can do the same.
- **ConvLSTM fabricates missing timesteps (Phase 4).** `trend_engine.py` pads short sequences by dilating the last mask — synthetic growth is fed to the model as observation.
- **No held-out evaluation exists (Phase 2).** All checkpoint metrics (e.g. the "93.7% accuracy") are training-set numbers; model selection is by training loss; there is no event-level split or test set.
- **Risk fusion diverges from PRD §9.5 (Phase 4 — scope decision).** A 0.20-weight ML-confidence term replaces the documented weights; ADR-002's "no trained ML in the critical path" is not currently enforced.
- **Weak-label "shadow" class is unreachable (Phase 2).** In `train_segformer.py`, water is assigned all VV < −22 dB, so shadow (VV < −25, not water) can never be labeled.
- **Docker weights-path divergence (Phase 3).** The engine's default checkpoint path resolves to `/data/processed` in the container while the pipeline mounts `/app/data/processed` — a deployed container silently runs the fallback despite mounted checkpoints.
