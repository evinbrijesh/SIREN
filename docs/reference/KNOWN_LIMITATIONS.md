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

- **Sentinel-1 swath coverage.** The available ascending-orbit S1 pair covers only the western AOI; the Imja lake proper (86.925°E) is outside the swath. The demo uses prepared scenario masks near Imja for observations 2–3 (clearly labeled in `detect/scenario.py`). The SAR pipeline code is real and validated on the covered region.
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

### Pipeline observation acceptance (Phase 4 blocker)
`run_pipeline()` accepts only the three hardcoded demo observation IDs (`obs-001`, `obs-002`, `obs-003`). Registering a new satellite product in the database does not make it processable. A live acquisition service can download and store real products, but the frozen pipeline will reject them with `ValueError: Unknown observation`. This must be resolved by an explicit scope decision before automatic live scoring is possible.

### Ingest scripts exit 0 on failure (Phase 1)
All four ingest scripts (`cdse.py`, `srtm.py`, `imerg.py`, `overpass.py`) return exit code 0 for many failure conditions. A scheduler that checks exit codes for success will silently miss failed downloads. Failure must produce a non-zero exit and a durable failure record.

### CDSE uses deprecated endpoint (Phase 1)
`backend/siren/ingest/cdse.py` targets `catalogue.dataspace.copernicus.eu/stac/search`. Current Copernicus documentation identifies `stac.dataspace.copernicus.eu/v1/search` as the active endpoint. The legacy endpoint is marked for deprecation and may stop returning results without warning.

### CDSE downloads entire archive into memory (Phase 1)
The download loop reads the full response body before writing. A Sentinel-1 GRD archive is approximately 1.7 GB. This will exhaust memory on a typical Lambda or small container. Downloads must stream to a temporary file.

### Overpass query missing river geometry (Phase 1, ADR-005 blocker)
`backend/siren/ingest/overpass.py` does not query `waterway=river` or `waterway=stream`. A live Overpass refresh would produce an extract without river geometry, breaking the corridor module's OSM river selection step. The committed `osm_infrastructure.geojson` was manually prepared and contains rivers; automated refresh would remove them.

### Overpass output format incompatible with corridor module (Phase 1)
The script emits `properties.tags` (nested). The corridor module expects flat properties (`waterway`, `highway`, `@id`). Automated refresh with the current script would produce a structurally incompatible extract that silently produces no corridor results.

### `openmeteo.py` date window walks forward, not backward (Phase 1)
The `_days_before(obs_date, -offset)` call in `openmeteo.py` moves the window forward in time. The seven-day antecedent rainfall accumulation is computed over the wrong dates. Missing precipitation is filled with zero. This makes the weather feature unreliable for any observation where the window does not coincide with available data.

### No job ledger or idempotency constraints (Phase 1)
There is no durable record of download attempts. A scheduler retry creates a duplicate `observations` row. The count-based run ID (`SELECT COUNT(*) + 1`) produces collisions under concurrent requests. Both must be fixed before any multi-instance deployment.

### SRTM uses outdated access URL (Phase 1)
`backend/siren/ingest/srtm.py` uses the old LP DAAC Data Pool URL format. NASA has migrated SRTM access to Earthdata Cloud with updated endpoints and requires Earthdata authentication for downloads. The current URL may return 404 or redirect incorrectly.

### Downloaded Sentinel-1 pair misses Imja Lake (known, documented)
The two downloaded Sentinel-1 archives (orbit 85, ascending) do not cover Imja Lake at 86.925°E. Scenario masks near Imja are used for demo observations 2 and 3. For operational monitoring, orbit 12 (ascending) and orbit 121 (descending) are the covering tracks, each with a 12-day repeat. The downloaded orbit-85 pair should not be used for Imja assessments.

### Population defaults are fabricated (Phase 2)
Unknown assets are inserted with `population = 1,240` (a hardcoded default). There is no measured or authority-verified population figure for most exposed settlements. The OSM extract has one `population` field across 1,100 features, and no `survey:date` or `check_date` tags. Exposure reports must label population figures with their source and confidence.

### SQLite not suitable for multi-instance hosted operation (Phase 3, ADR-001 addendum)
The current in-memory-default SQLite configuration, count-based IDs, and WAL limitations are incompatible with a multi-process, multi-instance hosted service. See ADR-001 addendum and ADR-007.

### Review suppression logic incorrect (Phase 3)
Any historical `confirm` review qualifies a dispatch, even if a later `reject` or `postpone` was recorded. A confirmed-then-rejected alert can still be dispatched. The latest review decision must be checked.

### ntfy.sh delivery is browser-side and unverified (Phase 3, ADR-009)
Live alert delivery is a browser-side HTTP call. `"status": "sent"` in the database does not mean the message was delivered. Closing the browser tab or a network error during the ntfy POST silently loses the alert. A server-side delivery outbox with receipts and retry is required for operational use.

### Frontend automatic mock fallback on any API error (Phase 3)
The frontend API client falls back to mock data on any HTTP error, including server errors and authentication failures. In a live service, this can display stale or simulated assessments without any indication that the backend is unavailable.

### Audit `_audit()` call does not commit (Phase 3)
The final `_audit()` call in the pipeline does not commit the transaction. The last audit entry may be lost on connection close or crash, and can hold a writer lock on SQLite. The final audit event must be committed as part of the run-completion transaction.

### Audit hash verification uses recomputed hashes (Phase 3)
The `/audit` endpoint recomputes hashes from current field values rather than comparing against stored `prev_hash`/`event_hash`. This can conceal discrepancies in stored records rather than report them. Verification must use stored values.

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
