# ADR-013 Addendum — FNO Dual-Contract Operation

**Status:** ACCEPTED · **Date:** 2026-09-14 · **Parent:** [ADR-013](ADR-013-end-to-end-neural-pipeline.md) · **Amends:** [ADR-012](ADR-012-fno-hydrodynamic-surrogate.md)

---

## Context

ADR-012 established the FNO2D hydrodynamic surrogate with a frozen 2-channel input contract: `[DEM_norm, log(1+V_breach)/20.0]`. A trained checkpoint exists at `models/checkpoints/fno_hydro_surrogate_v1.pt` with `input_proj.weight: [32, 2]` (in_features=2, the scalar contract). This checkpoint passed ADR-012's validation (MAPE ≤ 20% on 2/3 events) and is the load-bearing hydrodynamic surrogate.

ADR-013 §9.7.3 introduces latent spatial conditioning: the FNO input expands from 2 channels to `2 + d_latent` to accept the projected segmentation bottleneck embedding `z_lake`. This changes the `input_proj` layer shape from `[width, 2]` to `[width, 2 + d_latent]`, making the existing checkpoint incompatible with the new contract.

A naive migration would either (a) break the existing checkpoint, or (b) silently fall back to random weights for the latent channels. Both are unacceptable for a safety-critical system.

## Decision

### Dual-contract operation

The FNO2D class supports **both contracts simultaneously** via the `in_channels` constructor parameter. No contract is deprecated; both coexist.

| Contract | `in_channels` | Input channels | Checkpoint | Status |
|---|---|---|---|---|
| **Scalar (ADR-012 baseline)** | 2 | `[DEM_norm, V_breach_scalar]` | `fno_hydro_surrogate_v1.pt` (`input_proj.weight: [32, 2]`) | **Frozen, load-bearing** |
| **Latent (ADR-013 experimental)** | 2 + d_latent | `[DEM_norm, V_breach_scalar, z_lake_0, ..., z_lake_{d_latent-1}]` | None yet — requires training on 500+ shallow-water sims | **Experimental, not load-bearing** |

### Rules

1. **The scalar contract (`in_channels=2`) is the default.** `FNO2D()` without arguments creates a scalar-contract model. The frozen checkpoint loads only into this contract.

2. **The latent contract (`in_channels=2+d_latent`) is experimental.** It may only be instantiated within isolated experimental runners (`tests/test_latent_coupling.py`, `ml/latent_coupling.py`, future training scripts). It must never be wired into the production pipeline (`pipeline.py`, `api/`) until ADR-013's E0 gate is evaluated.

3. **No silent contract migration.** Loading a scalar checkpoint into a latent-contract model must fail loudly (shape mismatch). Loading a latent checkpoint into a scalar-contract model must fail loudly. The `assert C == self.in_channels` check in `FNO2D.forward()` enforces this.

4. **The scalar contract is not deprecated.** Even after the latent contract passes its gate, the scalar contract remains as a fallback (per ADR-013's dual-track policy). The two contracts serve different purposes: scalar for the deterministic baseline path, latent for the neural-primary path.

5. **Checkpoint versioning.** Future latent-contract checkpoints must record `in_channels` in their metadata. The loader must verify `in_channels` matches before loading.

### Migration path (when E0 gate passes)

When the latent-conditioned FNO passes ADR-012's MAPE ≤ 20% on 2/3 events:

1. Train the latent-contract FNO on 500+ shallow-water simulations.
2. Evaluate on the ADR-012 held-out events.
3. If the gate passes, create a new ADR authorizing the latent contract as primary.
4. The scalar contract remains as a labeled fallback, not deprecated.
5. Update `pipeline.py` to use the latent contract, with the scalar contract as fallback.

## Consequences

- **Positive:** the existing frozen checkpoint is protected. No silent breakage. The experimental latent contract can be developed without risking the load-bearing path.
- **Negative:** two contracts must be maintained. Future FNO improvements must be applied to both (or the divergence must be documented). Test coverage must verify both contracts.
- **Risk:** if a developer accidentally instantiates `FNO2D(in_channels=18)` in the production pipeline, it will fail at load time (no checkpoint) — this is the desired behavior, not a risk.

## Verification

- `tests/test_latent_coupling.py::test_fno2d_scalar_only_backward_compat` — scalar contract works with default args.
- `tests/test_latent_coupling.py::test_fno2d_latent_conditioned` — latent contract accepts 2+d_latent channels.
- `tests/test_latent_coupling.py::test_fno2d_wrong_channels_raises` — wrong channel count raises AssertionError.
- `tests/test_latent_coupling.py::test_fno2d_existing_checkpoint_loads` — existing scalar checkpoint loads into `in_channels=2` model.
- `tests/test_latent_coupling.py::test_fno2d_latent_more_parameters` — latent contract has more parameters (wider input_proj).
