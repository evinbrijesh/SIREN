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
- **ML evidence layer (optional).** `ml/` provides a change-detection evidence layer with a deterministic fallback when torch is not installed. When torch is available, a Siamese U-Net / ChangeFormer path can be activated. The deterministic path is the default and always works (ADR-002 addendum).

## What's Simulated (Not Real at Runtime)

- **Alert channels.** SMS, LoRa, and Satellite dispatch are simulated — no real carriers, modems, or satellite messengers are contacted. The AuditView channel simulator shows delivery states for demo purposes.
- **Live satellite ingestion.** All scenes are pre-downloaded to `data/raw/`. The `ingest/` scripts can fetch from CDSE/Earthdata/Overpass, but the runtime demo makes zero network calls (ADR-004: offline-first).
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
- **Review latency:** Depends entirely on the human coordinator. SIREN does not auto-escalate on timeout — that's a policy decision for the deploying authority.

## What SIREN Does NOT Do (PRD §14)

- Predict the exact time of glacial-lake collapse
- Guarantee exact flood depth or flow path
- Diagnose specific diseases from satellite imagery
- Issue autonomous evacuation orders
- Guarantee delivery to every person in an area
- Replace government early-warning systems
- Identify individuals, locate trapped persons, or handle family reunification (V4 roadmap, requires authority partnership)

---

**Bottom line for judges:** The detection, corridor, exposure, scoring, and audit chain is real code on real data. The dispatch channels and live ingestion are simulated — SIREN is a decision-support and resilience layer, not a replacement for emergency infrastructure. The system deploys via Docker Compose (`./start.sh`) for a one-command demo.
