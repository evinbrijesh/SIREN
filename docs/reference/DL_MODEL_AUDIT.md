# SIREN — DL / Remote-Sensing Model Audit

**Date:** 2026-09-07 · **Status:** Completed audit (no code or weights changed) · **Scope:** the four PRD-nominated models (Siamese U-Net, ChangeFormer, SegFormer, ConvLSTM) — task fit, data adequacy, evaluation validity, ADR-002 integration compliance
**Companions:** [ADR-002](../adr/ADR-002-deterministic-first-ml.md) (addendum 2), [ADR-010](../adr/ADR-010-ml-evidence-isolation-and-retraining-path.md) (proposed response), [PRODUCTION_ML_PLAN.md](PRODUCTION_ML_PLAN.md) (what to build instead)

**Bottom line: no existing checkpoint is qualified for live hazard assessment.** The problems are mismatched training/inference inputs, synthetic supervision, no held-out evaluation, and integration paths that can suppress rule-based evidence — not model choice alone.

---

## 0. Executive verdicts

| Candidate | Verdict | Primary reason |
|---|---|---|
| Siamese U-Net | **Keep as a later candidate; replace the immediate experiment** with a single-date SAR water segmenter | Right task family, but training synthesizes "before" images from the labels, and runtime feeds binary masks instead of SAR σ0 |
| ChangeFormer | **Drop from the near-term roadmap** | Not implemented; released pretrained checkpoints are optical building-change (LEVIR-CD / DSIFN-CD), not SAR water; no advantage over a smaller baseline under current constraints |
| "SegFormer" | **Drop the current crop classifier from operational filtering** | Not the SegFormer architecture (a patch-embed + one attention block + global pooling crop classifier); five-class supervision is threshold-generated; the "shadow" weak-label class is unreachable (labeling bug); component-level suppression can remove rule-detected evidence |
| ConvLSTM | **Replace with deterministic time-aware trend estimation** | Trained on synthetic mask progressions; no elapsed-time input; inference fabricates missing timesteps by dilating the last mask |

---

## 1. Claimed vs. implemented

| Claim (PRD §6.3/§9.2/§9.3, ADR-002 addendum) | Implemented reality |
|---|---|
| Siamese U-Net / ChangeFormer inference path | Siamese U-Net exists (`ml/model.py`, `ml/engine.py`); **ChangeFormer is not implemented at all** |
| SegFormer classifies changed pixels into functional classes | `ml/model.py::SegFormerHead` is a **crop classifier** (one class per 64×64 crop), not a dense segmenter; `ml/segformer_engine.py` classifies connected-component crops |
| ConvLSTM is a research-grade V3 item | A ConvLSTM hybrid (**Stage 4**) is implemented and **can replace the configured trend class** that feeds the hazard score |
| ML is never the sole source of a score | A 0.20-weight ML-confidence term is inside the H formula; the trend term can be model-driven; the crop classifier can delete rule-detected pixels from the evidence mask |
| PRD §9.5 weights 0.30/0.25/0.20/0.15/0.10 | `risk/fusion.py` implements 0.25/0.20/0.15/0.10/0.10/**0.20(ML)** |

## 2. Task–model fit (condensed)

- **Siamese U-Net** — designed for bi-temporal change detection on **co-registered pairs** ([Daudt et al. 2018](https://arxiv.org/abs/1810.08462)); works with binary change labels; SAR input is architecturally fine (2-ch VV/VH) but requires SAR-consistent training/preprocessing. SAR-before/optical-after is cross-modal and not a drop-in. The current implementation uses absolute feature differences → symmetric; it cannot distinguish expansion from recession by itself.
- **ChangeFormer** — transformer Siamese **binary change detection** ([Bandara & Patel 2022](https://arxiv.org/abs/2201.01293)); pretrained releases are optical (LEVIR/DSIFN). No SAR-water value under current constraints.
- **SegFormer** — **single-image semantic segmentation** ([Xie et al. 2021](https://arxiv.org/abs/2105.15203)); valid adaptations are per-date water mapping + differencing, or a trained change head. The implemented crop classifier is neither. SegFormer code is research/evaluation-licensed — review before commercial use.
- **ConvLSTM** — introduced for **precipitation nowcasting** on regular radar sequences ([Shi et al. 2015](https://arxiv.org/abs/1506.04214v1)). SIREN's cadence is irregular (12-day repeats, two orbits, weather-dependent optical); a vanilla ConvLSTM has **no elapsed-time input**, so "rapidly" is not anchored in time. The router does not break it per se — but the model never sees modality, orbit, or Δt.

**Best fit among the four:** Siamese U-Net (task family). **Best fit for the data actually available:** a compact **single-date SAR water-segmentation U-Net** + deterministic bi-temporal differencing.

## 3. Training-data adequacy findings

1. **Siamese training leaks the label into the input.** `train.py` builds `t0` by replacing labeled water with median land backscatter; outside labeled water the pair is identical. The model can learn the replacement operation, not real temporal SAR behavior. Permanent water is treated as newly-appearing water. Invalid labels (`-1`) become background. (`train.py` `Sen1Floods11Dataset.__getitem__`)
2. **Weak labels are threshold echoes, and one class is unreachable.** `train_segformer.py::_assign_weak_labels` assigns water to *all* valid pixels with VV < −22 dB, then requires shadow to be VV < −25 **and not water** — so **no valid pixel can ever be labeled shadow** (verified by synthetic probe). The "debris/snowmelt" classes are single-threshold inventions, not physical labels.
3. **ConvLSTM trains on generated mask progressions.** `train_temporal.py` synthesizes expanding/fluctuating sequences from single chips; "stable" is an exact repeated mask. The model can learn the generator, not hydrology.
4. **The local basin scenes (2 S1 + 1 S2) cannot train anything from scratch** and the downloaded orbit-85 pair misses Imja. The **local Sen1Floods11 train manifest (252 pairs)** does support a legitimate single-date water-segmentation experiment — with the dataset's **official event-level splits**, which the current training does not use (it trains on `split="train"` only).
5. **Checkpoint metadata are training-set numbers, not test results** (inspected without training):

| Checkpoint | Saved metadata |
|---|---|
| `siamese_unet_weights.pt` | epoch 50, loss ≈ 0.049, in_channels=2 |
| `segformer_classifier_weights.pt` | epoch 49, accuracy ≈ 61.8%, weak-labeled |
| `convlstm_trend_weights.pt` | epoch 75, accuracy ≈ 93.7% |

   All three training loops use `split="train"`, select by training loss, and have **no validation or test evaluation anywhere in the codebase**.

## 4. Evaluation methodology requirements

- **Split by event/region before cropping or augmenting**: all patches from one acquisition/event, both halves of a pair, overlapping tiles, and all synthetic descendants must stay together. Random 80/20 chip splits are invalid.
- **Two separate claims need separate evidence**: "works on future Imja dates" (chronological untouched Imja test set) vs. "generalizes to other basins" (entire held-out basins/events).
- **Protocol:** train on selected external events → select on separate events → freeze everything → test on held-out events → prospective test on unseen basin dates. If basin data is used for fine-tuning, reserve new untouched dates and label it adaptation, not zero-shot.
- **Causality for temporal evaluation:** only imagery/weather available by the assessment time.
- **Metrics:** water/change IoU, precision, recall, F1 (not pixel accuracy); permanent-vs-new-water errors; water-area and shoreline error in physical units; per-orbit/season/terrain breakdowns; false alarms per acquisition; missed critical assets; calibration. Report event-level uncertainty.
- **Required ablations:** rules-only vs. ML-only (offline comparator) vs. rules+ML-auxiliary vs. model-unavailable. Any claimed ML value must survive these.
- Water-segmentation accuracy must never be reported as GLOF prediction accuracy or evacuation lead time.

## 5. Integration findings (ADR-002 compliance)

1. **Runtime feeds the model the wrong data.** The trained model expects normalized VV/VH SAR; the pipeline feeds `baseline_water_mask.tif` and a scenario expansion mask, resized by array shape (ignoring transforms) with channel replication. A tensor that fits is not a valid input. (`pipeline.py::_try_ml_evidence_layer`)
2. **The consensus mask preserves rule positives — then the next stage can delete them.** `consensus.py` correctly includes rule-based detections, but `segformer_engine.classify_change_crops` replaces the consensus with `filtered_mask`, deleting whole connected components labeled shadow/snowmelt — **including rule-detected ones**, even via the deterministic fallback (synthetic probe: 100 rule pixels → 0 after fallback filtering). Mitigating fact: the main corridor and `change_mask_uri` still read the original scenario mask — so the system has **inconsistent evidence paths**, not full ML replacement.
3. **Trend can be model-replaced.** `run_pipeline()` overwrites the configured `trend_class` with the hybrid ConvLSTM result before risk fusion; `_deterministic_fallback`'s output also flows into H.
4. **Fusion weight divergence** (see §1 table); the "ML confidence" is agreement-constant-derived, not calibrated probability; it is averaged differently in ML vs. fallback paths; it is computed before Stage-2 filtering and never recomputed; fallback fabricates a synthetic confidence map.
5. **Temporal fabrication:** `trend_engine.classify_trend` pads short sequences by dilating the last mask (synthetic growth fed to the model as observation; probe: one 100-px mask became 100→140→184→232). Truncation silently drops older observations.
6. **Failure paths:** missing-torch and generic exceptions are caught (verified with an injected RuntimeError); but a failed weight load can still construct a fresh ImageNet-pretrained encoder; `classify_temporal_trend`'s except-branches re-instantiate the engine (second failure possible); no process isolation from OS-level OOM kills; no input-contract/finite-value validation; the "tiled inference" docstring is unimplemented (whole-array forward pass = memory risk).
7. **Deployment path bug:** in the Docker layout the engine's default weights path resolves to `/data/processed` while the pipeline uses `/app/data/processed` — a deployed container silently runs the fallback despite mounted checkpoints.
8. **Registry accuracy:** `registry.py` reports "loaded" from file-existence + torch importability, not actual loadability; descriptions repeat the inaccurate "Siamese U-Net / ChangeFormer" and "MiT-B0" naming.

## 6. Verification scope and limits

- Reviewed: PRD v4.5, ADR-002/003, all of `backend/siren/ml/` (model/engine/train/registry/consensus/segformer_engine/trend_engine/temporal), `pipeline.py`, `risk/fusion.py`, `tests/test_ml.py`, `pyproject.toml`, `Dockerfile.backend`, checkpoint metadata, Sen1Floods11 local manifest, and primary sources for each architecture/dataset.
- Ran: 12 targeted software tests (all passed — shapes/fallback/consensus behavior, not accuracy); isolated synthetic probes for weak-labeling, filtered-mask override, temporal padding, and exception handling.
- **Not done:** no training, no full real-data pipeline runs, no held-out accuracy measurement, no file modifications.
