"""Demo scenario masks (Roadmap fallback — clearly labeled).

The available ascending-orbit S1 pair does NOT cover the Imja lake (swath
edge ~86.69°E at lat 27.9; the lake is at 86.925°E — see sar.py). The demo
scenario therefore uses PREPARED expansion masks near the Imja lake,
generated here. The SAR pipeline itself (sar.py) is real and validated on
the covered western AOI.

Scenario (PRD §16 retrospective what-if):
  - obs-001 (2026-07-23): +8% supraglacial pond expansion  -> Advisory
  - obs-002 (2026-08-04): +28% expansion (disaster day)    -> Elevated/Critical

The masks are generated deterministically (seeded) around the Imja lake
centroid, expanding outward — physically plausible moraine-breach geometry.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Imja lake centroid (lon, lat) — the demo change source
IMJA_CENTROID = (86.925, 27.895)
# Scenario expansion percentages (PRD §16)
SCENARIO_EXPANSIONS = {"obs-001": 0.08, "obs-002": 0.28}
# Grid: 200x200 cells at ~30m spanning ~6km around the lake
GRID_SIZE = 200
CELL_M = 30.0
RNG_SEED = 42  # Hard Rule 6: reproducibility


def scenario_expansion_mask(expansion_pct: float, seed: int = RNG_SEED) -> tuple[np.ndarray, dict]:
    """Generate a deterministic water-expansion mask for the demo scenario.

    The baseline lake is an irregular blob; the expansion grows it by
    `expansion_pct` with a ragged (moraine-breach-like) edge on the
    downstream side.

    Returns (mask, meta). mask is a boolean array; meta carries the transform
    and scenario metadata.
    """
    rng = np.random.default_rng(seed)
    n = GRID_SIZE
    cy, cx = n // 2, n // 2

    # Baseline lake: irregular ellipse via low-frequency polar noise
    theta = np.linspace(0, 2 * np.pi, 720)
    base_r = 25 * (
        1.0
        + 0.15 * np.sin(3 * theta + rng.uniform(0, 2 * np.pi))
        + 0.10 * np.sin(5 * theta + rng.uniform(0, 2 * np.pi))
    )

    yy, xx = np.mgrid[0:n, 0:n]
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)

    # Baseline mask: inside the irregular ellipse
    r_at_ang = np.interp(ang % (2 * np.pi), theta, base_r)
    baseline = r <= r_at_ang
    base_px = int(baseline.sum())

    # Expansion: ragged growth band on the downstream (south-west) side,
    # sized to hit the target percentage.
    target_px = int(base_px * (1 + expansion_pct))
    growth_pref = (np.cos(ang - np.deg2rad(215)) + 1) / 2  # SW-facing bias
    noise = rng.uniform(0, 1, size=baseline.shape)
    # Expand cells just outside the baseline, biased to the SW, deterministic
    band = (~baseline) & (r <= r_at_ang.max() + 12)
    score = growth_pref * 0.7 + noise * 0.3
    order = np.argsort(np.where(band, score, -1).ravel())[::-1]
    need = target_px - base_px
    flat_mask = baseline.ravel().copy()
    flat_mask[order[:need]] = True
    expanded = flat_mask.reshape(baseline.shape)

    # GeoTransform: center the grid on the Imja centroid
    span = n * CELL_M
    transform = from_bounds(
        IMJA_CENTROID[0] - span / 2 / 111_320 / np.cos(np.deg2rad(IMJA_CENTROID[1])) * 1.0,
        IMJA_CENTROID[1] - span / 2 / 111_320,
        IMJA_CENTROID[0] + span / 2 / 111_320 / np.cos(np.deg2rad(IMJA_CENTROID[1])) * 1.0,
        IMJA_CENTROID[1] + span / 2 / 111_320,
        n, n,
    )

    meta = {
        "scenario": True,
        "expansion_pct": expansion_pct,
        "baseline_px": base_px,
        "expanded_px": int(expanded.sum()),
        "actual_expansion_pct": round((int(expanded.sum()) / base_px - 1) * 100, 1),
        "centroid": IMJA_CENTROID,
        "seed": seed,
        "transform": transform,
    }
    return expanded, meta


def write_scenario_masks(out_dir: str = "data/processed") -> dict:
    """Generate and write both scenario masks (+8%, +28%) as GeoTIFFs."""
    results = {}
    for obs_id, pct in SCENARIO_EXPANSIONS.items():
        mask, meta = scenario_expansion_mask(pct)
        out = f"{out_dir}/{obs_id}_expansion_mask.tif"
        with rasterio.open(
            out, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
            count=1, dtype="uint8", crs="EPSG:4326", transform=meta["transform"],
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)
        results[obs_id] = {"path": out, **{k: v for k, v in meta.items() if k != "transform"}}
    return results


if __name__ == "__main__":
    for obs_id, info in write_scenario_masks().items():
        print(f"✓ {obs_id}: {info['path']} — {info['actual_expansion_pct']}% expansion "
              f"({info['baseline_px']} -> {info['expanded_px']} px)")