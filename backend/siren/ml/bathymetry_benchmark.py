"""Lake volume estimation and leave-one-lake-out benchmark (ADR-013 §9.7.1).

This module provides volume estimation from surveyed bathymetry points and
a benchmark comparing three methods:

    1. Huggel et al. (2002) empirical area-volume power law (the current
       production baseline in breach_volume.py):
           V = 0.104 * A^1.421  (A in m², V in m³)

    2. Surveyed mean-depth integration (ground truth proxy):
           V = mean_depth * lake_area
       where mean_depth is the arithmetic mean of surveyed depth points
       and lake_area is from the outline polygon (4-lake dataset) or the
       convex hull of surveyed points (16-lake dataset, proxy).

    3. Fitted power-law regression (data-driven baseline):
           V = a * A^b
       fitted on the training lakes and evaluated on the held-out lake.

The benchmark runs leave-one-lake-out cross-validation across all 20
surveyed lakes and reports MAPE (Mean Absolute Percentage Error) for
each method. The Huggel formula is the gate: any neural model must
beat its LOO MAPE to be considered for promotion.

Ground truth volumes come from two sources:
    - The global compilation (Zhang & Wang 2023) for 6 Himalayan lakes
      with published volumes (Bechung Tsho, Bielongco, Galongco,
      Guangxieco, Luggye Tsho, Raphstreng Tsho).
    - The 4-lake dataset's published volumes (Das & Ramsankaran 2025):
      Kya Tso 0.89 MCM, Panchi Nala 0.44 MCM, Gepang Gath 24.12 MCM,
      Samudra Tapu 24.69 MCM.
    - For lakes without published volumes, the surveyed mean-depth ×
      area is used as a self-consistent ground truth (not independent,
      but useful for LOO comparison).

Scientific notes:
    - The convex hull of surveyed points underestimates lake area (points
      don't cover the shoreline). This biases the surveyed volume low.
      The 4-lake dataset has real outlines, so its volume estimates are
      more reliable.
    - The Huggel formula was calibrated on Himalayan glacial lakes and is
      the current production baseline. Beating it requires either more
      data or a better model.
    - The power-law regression is the simplest data-driven baseline. If
      it can't beat Huggel on LOO, a neural model is unlikely to either
      with only 20 lakes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import geopandas as gpd
from scipy.optimize import curve_fit

from siren.ml.bathymetry_dataset import (
    LakeRecord,
    load_all_surveyed_lakes,
    load_global_compilation,
    generate_loo_splits,
    LakeSplit,
)
from siren.risk.breach_volume import HUGGEL_ALPHA, HUGGEL_GAMMA

logger = logging.getLogger(__name__)

# Published volumes for the 4-lake dataset (Das & Ramsankaran 2025, Zenodo)
# Values in millions of cubic metres (MCM = 1e6 m³)
DAS_PUBLISHED_VOLUMES_MCM: dict[str, float] = {
    "Kya Tso Lake": 0.89,
    "Panchi Nala Lake": 0.44,
    "Gepang Gath Lake": 24.12,
    "Samudra Tapu Lake": 24.69,
}


# ---------------------------------------------------------------------------
# Lake area estimation
# ---------------------------------------------------------------------------


def _published_entries_for_dense_lakes() -> list:
    """Compilation entries eligible as published values for the dense
    surveyed lakes.

    The 20 dense-sounding lakes are all proglacial moraine-dammed lakes
    whose published metadata lives in the "proglacial" worksheet. Other
    sheets contain same-named lakes measured by different programmes
    (e.g. Bencoguoco's periglacial entry), so lookups are restricted to
    proglacial rows.
    """
    return [e for e in load_global_compilation() if e.lake_type == "proglacial"]


def _get_published_area_km2(record: LakeRecord) -> float | None:
    """Get the published area for a lake from the global compilation.

    Args:
        record: LakeRecord.

    Returns:
        Published area in km², or None if not available.
    """
    entries = _published_entries_for_dense_lakes()
    for entry in entries:
        name_key = entry.name.lower().strip()
        rec_key = record.lake_name.lower().strip()
        if name_key == rec_key or name_key == rec_key.replace(" tsho", "").replace(" lake", ""):
            if entry.area_km2 is not None:
                return entry.area_km2
    return None


def compute_lake_area_km2(record: LakeRecord) -> float:
    """Compute lake surface area in km² from the best available source.

    Priority:
        1. Published area from the global compilation (most accurate for
           the 16-lake dataset — the convex hull of points is unreliable).
        2. Outline polygon area (for the 4-lake dataset).
        3. Convex hull of surveyed points (fallback proxy — may under- or
           over-estimate because points don't cover the shoreline).

    Args:
        record: LakeRecord with points and optional outline.

    Returns:
        Lake area in km².
    """
    # 1. Published area from the global compilation
    published_area = _get_published_area_km2(record)
    if published_area is not None:
        return published_area

    # 2. Outline polygon area (4-lake dataset)
    if record.outline is not None:
        return float(record.outline.area) / 1e6

    # 3. Convex hull of surveyed points (fallback)
    if record.crs.to_epsg() == 4326:
        # WGS84 — need to reproject to UTM
        minx = record.bounds[0]
        utm_zone = int((minx + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if record.bounds[1] >= 0 else f"EPSG:327{utm_zone:02d}"
        points_utm = record.points.to_crs(utm_crs)
    else:
        points_utm = record.points

    hull = points_utm.geometry.union_all().convex_hull
    return float(hull.area) / 1e6


# ---------------------------------------------------------------------------
# Volume estimation methods
# ---------------------------------------------------------------------------


def huggel_volume_m3(area_km2: float) -> float:
    """Huggel et al. (2002) area-volume power law.

    V = 0.104 * A^1.421  (A in m², V in m³)

    Args:
        area_km2: lake surface area in km².

    Returns:
        Volume in m³.
    """
    area_m2 = area_km2 * 1e6
    return float(HUGGEL_ALPHA * (area_m2 ** HUGGEL_GAMMA))


def surveyed_volume_m3(record: LakeRecord, area_km2: float) -> float:
    """Volume from surveyed mean depth × lake area.

    V = mean_depth * area

    This is a ground-truth proxy. For lakes with published volumes, the
    published value should be preferred. For lakes without, this provides
    a self-consistent estimate.

    Args:
        record: LakeRecord with surveyed depth points.
        area_km2: lake surface area in km².

    Returns:
        Volume in m³.
    """
    mean_depth = record.mean_depth_m
    area_m2 = area_km2 * 1e6
    return float(mean_depth * area_m2)


def get_published_volume_m3(record: LakeRecord) -> float | None:
    """Get the published volume for a lake, if available.

    Checks the global compilation and the 4-lake dataset's published values.

    Args:
        record: LakeRecord.

    Returns:
        Published volume in m³, or None if not available.
    """
    # Check 4-lake published volumes
    if record.lake_name in DAS_PUBLISHED_VOLUMES_MCM:
        return DAS_PUBLISHED_VOLUMES_MCM[record.lake_name] * 1e6

    # Check global compilation
    entries = _published_entries_for_dense_lakes()
    for entry in entries:
        name_key = entry.name.lower().strip()
        rec_key = record.lake_name.lower().strip()
        if name_key == rec_key or name_key == rec_key.replace(" tsho", "").replace(" lake", ""):
            if entry.volume_mcm is not None:
                return entry.volume_mcm * 1e6

    return None


# ---------------------------------------------------------------------------
# Power-law regression
# ---------------------------------------------------------------------------


def _power_law(A: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power-law function: V = a * A^b."""
    return a * np.power(A, b)


def fit_power_law(
    areas_km2: np.ndarray,
    volumes_m3: np.ndarray,
    initial_guess: tuple[float, float] = (HUGGEL_ALPHA, HUGGEL_GAMMA),
) -> tuple[float, float]:
    """Fit a power-law V = a * A^b to area-volume data.

    Uses log-space linear regression for numerical stability across the
    wide range of lake areas (0.08–5.5 km²) and volumes (0.4–375 MCM).
    The log-transform linearises the power law:
        log(V) = log(a) + b * log(A)
    and ordinary least squares on the log-transformed data gives stable
    estimates even when curve_fit on the raw data fails.

    Args:
        areas_km2: array of lake areas in km².
        volumes_m3: array of lake volumes in m³.
        initial_guess: initial parameters (a, b) — used as a fallback if
            log-space fitting fails.

    Returns:
        Fitted parameters (a, b).
    """
    # Filter to positive values (log requires > 0)
    mask = (areas_km2 > 0) & (volumes_m3 > 0)
    areas_m2 = areas_km2[mask] * 1e6
    vols = volumes_m3[mask]

    if len(areas_m2) < 2:
        return initial_guess

    # Log-space linear regression: log(V) = log(a) + b * log(A)
    log_A = np.log(areas_m2)
    log_V = np.log(vols)

    # OLS fit
    n = len(log_A)
    b_fit = float(np.sum((log_A - log_A.mean()) * (log_V - log_V.mean())) /
                  np.sum((log_A - log_A.mean()) ** 2))
    log_a_fit = float(log_V.mean() - b_fit * log_A.mean())
    a_fit = float(np.exp(log_a_fit))

    if not (np.isfinite(a_fit) and np.isfinite(b_fit)):
        return initial_guess

    return a_fit, b_fit


def predict_power_law_volume_m3(
    area_km2: float,
    params: tuple[float, float],
) -> float:
    """Predict volume using fitted power-law parameters.

    Args:
        area_km2: lake area in km².
        params: fitted (a, b) parameters.

    Returns:
        Predicted volume in m³.
    """
    a, b = params
    area_m2 = area_km2 * 1e6
    return float(a * (area_m2 ** b))


# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------


@dataclass
class LakeVolumeEstimate:
    """Volume estimate for a single lake.

    Attributes:
        lake_id: lake identifier.
        lake_name: lake name.
        area_km2: lake surface area used for estimation.
        mean_depth_m: surveyed mean depth.
        max_depth_m: surveyed max depth.
        huggel_volume_m3: Huggel formula volume.
        surveyed_volume_m3: surveyed mean-depth × area volume.
        published_volume_m3: published volume from literature, or None.
        ground_truth_volume_m3: the best available ground truth volume.
        ground_truth_source: "published" or "surveyed".
    """

    lake_id: str
    lake_name: str
    area_km2: float
    mean_depth_m: float
    max_depth_m: float
    huggel_volume_m3: float
    surveyed_volume_m3: float
    published_volume_m3: float | None
    ground_truth_volume_m3: float
    ground_truth_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_id": self.lake_id,
            "lake_name": self.lake_name,
            "area_km2": round(self.area_km2, 4),
            "mean_depth_m": round(self.mean_depth_m, 2),
            "max_depth_m": round(self.max_depth_m, 2),
            "huggel_volume_m3": round(self.huggel_volume_m3, 1),
            "surveyed_volume_m3": round(self.surveyed_volume_m3, 1),
            "published_volume_m3": (
                round(self.published_volume_m3, 1)
                if self.published_volume_m3 is not None
                else None
            ),
            "ground_truth_volume_m3": round(self.ground_truth_volume_m3, 1),
            "ground_truth_source": self.ground_truth_source,
        }


@dataclass
class LOOBenchmarkResult:
    """Result of a single leave-one-lake-out benchmark fold.

    Attributes:
        test_lake_id: the held-out lake.
        test_lake_name: the held-out lake name.
        ground_truth_volume_m3: ground truth volume.
        huggel_volume_m3: Huggel prediction.
        huggel_ape: Huggel absolute percentage error.
        regression_volume_m3: fitted power-law prediction.
        regression_ape: regression absolute percentage error.
        regression_params: fitted (a, b) parameters.
    """

    test_lake_id: str
    test_lake_name: str
    ground_truth_volume_m3: float
    huggel_volume_m3: float
    huggel_ape: float
    regression_volume_m3: float
    regression_ape: float
    regression_params: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_lake_id": self.test_lake_id,
            "test_lake_name": self.test_lake_name,
            "ground_truth_volume_m3": round(self.ground_truth_volume_m3, 1),
            "huggel_volume_m3": round(self.huggel_volume_m3, 1),
            "huggel_ape": round(self.huggel_ape, 2),
            "regression_volume_m3": round(self.regression_volume_m3, 1),
            "regression_ape": round(self.regression_ape, 2),
            "regression_params": list(self.regression_params),
        }


@dataclass
class BenchmarkSummary:
    """Summary of the full LOO benchmark.

    Attributes:
        n_lakes: number of lakes in the benchmark.
        n_with_published: number of lakes with published volumes.
        huggel_mape: Huggel formula MAPE across all folds.
        regression_mape: fitted power-law regression MAPE across all folds.
        huggel_median_ape: Huggel median APE.
        regression_median_ape: regression median APE.
        folds: per-fold results.
        gate_target_mape: target MAPE for the E2 gate (15%).
        huggel_passes_gate: whether Huggel passes the gate.
        regression_passes_gate: whether regression passes the gate.
    """

    n_lakes: int
    n_with_published: int
    huggel_mape: float
    regression_mape: float
    huggel_median_ape: float
    regression_median_ape: float
    folds: list[LOOBenchmarkResult] = field(default_factory=list)
    gate_target_mape: float = 0.15
    huggel_passes_gate: bool = False
    regression_passes_gate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_lakes": self.n_lakes,
            "n_with_published": self.n_with_published,
            "huggel_mape": round(self.huggel_mape, 4),
            "regression_mape": round(self.regression_mape, 4),
            "huggel_median_ape": round(self.huggel_median_ape, 4),
            "regression_median_ape": round(self.regression_median_ape, 4),
            "gate_target_mape": self.gate_target_mape,
            "huggel_passes_gate": self.huggel_passes_gate,
            "regression_passes_gate": self.regression_passes_gate,
            "folds": [f.to_dict() for f in self.folds],
        }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def build_volume_estimates(
    records: list[LakeRecord] | None = None,
) -> list[LakeVolumeEstimate]:
    """Build volume estimates for all surveyed lakes.

    For each lake, computes:
        - Lake area (from outline or convex hull of points)
        - Huggel volume
        - Surveyed mean-depth × area volume
        - Published volume (if available)
        - Ground truth volume (published if available, else surveyed)

    Args:
        records: list of LakeRecord objects. Loaded if None.

    Returns:
        List of LakeVolumeEstimate objects.
    """
    if records is None:
        records = load_all_surveyed_lakes()

    estimates: list[LakeVolumeEstimate] = []
    for record in records:
        area_km2 = compute_lake_area_km2(record)
        huggel_v = huggel_volume_m3(area_km2)
        surveyed_v = surveyed_volume_m3(record, area_km2)
        published_v = get_published_volume_m3(record)

        if published_v is not None:
            ground_truth = published_v
            source = "published"
        else:
            ground_truth = surveyed_v
            source = "surveyed"

        estimates.append(LakeVolumeEstimate(
            lake_id=record.lake_id,
            lake_name=record.lake_name,
            area_km2=area_km2,
            mean_depth_m=record.mean_depth_m,
            max_depth_m=record.max_depth_m,
            huggel_volume_m3=huggel_v,
            surveyed_volume_m3=surveyed_v,
            published_volume_m3=published_v,
            ground_truth_volume_m3=ground_truth,
            ground_truth_source=source,
        ))

    return estimates


def run_loo_benchmark(
    records: list[LakeRecord] | None = None,
    estimates: list[LakeVolumeEstimate] | None = None,
    gate_target_mape: float = 0.15,
) -> BenchmarkSummary:
    """Run the leave-one-lake-out volume estimation benchmark.

    For each fold:
        1. Hold out one lake as the test set.
        2. Fit a power-law V = a * A^b on the remaining 19 lakes using
           their ground truth volumes.
        3. Predict the held-out lake's volume using:
           a. Huggel formula (no fitting needed)
           b. Fitted power-law regression
        4. Compute APE for each method against the ground truth.

    Args:
        records: list of LakeRecord objects. Loaded if None.
        estimates: pre-computed volume estimates. Built if None.
        gate_target_mape: target MAPE for the E2 gate (default 15%).

    Returns:
        BenchmarkSummary with per-fold results and aggregate MAPE.
    """
    if estimates is None:
        estimates = build_volume_estimates(records)

    # Sort by lake_id for reproducibility
    estimates_sorted = sorted(estimates, key=lambda e: e.lake_id)

    folds: list[LOOBenchmarkResult] = []
    for i, test_est in enumerate(estimates_sorted):
        train_ests = [e for j, e in enumerate(estimates_sorted) if j != i]

        # Ground truth for the test lake
        gt = test_est.ground_truth_volume_m3

        # Huggel prediction (no fitting needed)
        huggel_v = huggel_volume_m3(test_est.area_km2)
        huggel_ape = abs(huggel_v - gt) / gt if gt > 0 else float("inf")

        # Fit power-law on training lakes
        train_areas = np.array([e.area_km2 for e in train_ests])
        train_volumes = np.array([e.ground_truth_volume_m3 for e in train_ests])

        try:
            params = fit_power_law(train_areas, train_volumes)
            reg_v = predict_power_law_volume_m3(test_est.area_km2, params)
        except Exception as e:
            logger.warning("Power-law fit failed for %s: %s", test_est.lake_name, e)
            params = (HUGGEL_ALPHA, HUGGEL_GAMMA)
            reg_v = huggel_v

        reg_ape = abs(reg_v - gt) / gt if gt > 0 else float("inf")

        folds.append(LOOBenchmarkResult(
            test_lake_id=test_est.lake_id,
            test_lake_name=test_est.lake_name,
            ground_truth_volume_m3=gt,
            huggel_volume_m3=huggel_v,
            huggel_ape=huggel_ape,
            regression_volume_m3=reg_v,
            regression_ape=reg_ape,
            regression_params=params,
        ))

    # Aggregate
    huggel_apes = [f.huggel_ape for f in folds]
    reg_apes = [f.regression_ape for f in folds]
    huggel_mape = float(np.mean(huggel_apes))
    reg_mape = float(np.mean(reg_apes))
    huggel_median = float(np.median(huggel_apes))
    reg_median = float(np.median(reg_apes))

    n_published = sum(1 for e in estimates if e.published_volume_m3 is not None)

    summary = BenchmarkSummary(
        n_lakes=len(estimates),
        n_with_published=n_published,
        huggel_mape=huggel_mape,
        regression_mape=reg_mape,
        huggel_median_ape=huggel_median,
        regression_median_ape=reg_median,
        folds=folds,
        gate_target_mape=gate_target_mape,
        huggel_passes_gate=huggel_mape < gate_target_mape,
        regression_passes_gate=reg_mape < gate_target_mape,
    )

    logger.info(
        "LOO benchmark complete: %d lakes (%d with published volumes), "
        "Huggel MAPE=%.1f%%, Regression MAPE=%.1f%%, gate=%.0f%%",
        summary.n_lakes,
        summary.n_with_published,
        huggel_mape * 100,
        reg_mape * 100,
        gate_target_mape * 100,
    )

    return summary


# ---------------------------------------------------------------------------
# Metadata-level benchmark over the global compilation
# ---------------------------------------------------------------------------


def _normalise_lake_name(name: str) -> str:
    """Normalise a compilation lake name for cross-year grouping."""
    return name.lower().strip()


def run_metadata_loo_benchmark(
    entries: list | None = None,
    gate_target_mape: float = 0.15,
) -> dict[str, Any]:
    """Grouped leave-one-lake-out benchmark over the global compilation.

    Evaluates the Huggel formula and a fitted power-law regression on the
    published (area, volume) pairs from the global bathymetry
    compilation — ~300 metadata entries across five lake-type sheets, a
    much larger sample than the 20 dense-sounding lakes (metadata-level
    only; no bed-elevation ground truth).

    Entries are grouped by normalised lake name: all survey-year rows of
    the same lake are held out together so a lake's other-year
    measurements can't leak into its own training fold.

    Args:
        entries: GlobalCompilationEntry list. Loaded if None.
        gate_target_mape: target MAPE for the E2 gate (default 15%).

    Returns:
        Dict with entry/lake counts, overall + per-type + Himalaya-subset
        MAPE for both methods, and per-fold results.
    """
    if entries is None:
        entries = load_global_compilation()

    # Only entries with both published area and volume are evaluable
    evaluable = [
        e for e in entries
        if e.area_km2 is not None and e.area_km2 > 0
        and e.volume_mcm is not None and e.volume_mcm > 0
    ]

    # Group by normalised name (same lake, different survey years)
    groups: dict[str, list] = {}
    for e in evaluable:
        groups.setdefault(_normalise_lake_name(e.name), []).append(e)
    group_names = sorted(groups)

    folds: list[dict[str, Any]] = []
    for hold_name in group_names:
        held = groups[hold_name]
        train = [e for n, g in groups.items() if n != hold_name for e in g]
        train_areas = np.array([e.area_km2 for e in train])
        train_volumes = np.array([e.volume_mcm * 1e6 for e in train])
        try:
            params = fit_power_law(train_areas, train_volumes)
        except Exception as exc:
            logger.warning("Power-law fit failed holding out %s: %s", hold_name, exc)
            params = (HUGGEL_ALPHA, HUGGEL_GAMMA)

        for e in held:
            gt = e.volume_mcm * 1e6
            huggel_v = huggel_volume_m3(e.area_km2)
            reg_v = predict_power_law_volume_m3(e.area_km2, params)
            folds.append({
                "lake_name": e.name,
                "lake_type": e.lake_type,
                "mountain": e.mountain,
                "survey_year": e.survey_year,
                "area_km2": e.area_km2,
                "ground_truth_volume_m3": round(gt, 1),
                "huggel_ape": abs(huggel_v - gt) / gt,
                "regression_ape": abs(reg_v - gt) / gt,
            })

    def _mape(subset: list[dict[str, Any]], key: str) -> float | None:
        return float(np.mean([f[key] for f in subset])) if subset else None

    def _summarise(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_entries": len(subset),
            "huggel_mape": _mape(subset, "huggel_ape"),
            "huggel_median_ape": (
                float(np.median([f["huggel_ape"] for f in subset])) if subset else None
            ),
            "regression_mape": _mape(subset, "regression_ape"),
            "regression_median_ape": (
                float(np.median([f["regression_ape"] for f in subset])) if subset else None
            ),
        }

    by_type: dict[str, dict[str, Any]] = {}
    for lake_type in sorted({f["lake_type"] for f in folds}):
        by_type[lake_type] = _summarise(
            [f for f in folds if f["lake_type"] == lake_type]
        )
    himalaya = [f for f in folds if "himalaya" in f["mountain"].lower()]

    overall = _summarise(folds)
    result = {
        "n_entries_total": len(entries),
        "n_entries_evaluable": len(evaluable),
        "n_unique_lakes": len(group_names),
        "gate_target_mape": gate_target_mape,
        "overall": overall,
        "himalaya_subset": _summarise(himalaya),
        "by_lake_type": by_type,
        "huggel_passes_gate": (
            overall["huggel_mape"] is not None
            and overall["huggel_mape"] < gate_target_mape
        ),
        "regression_passes_gate": (
            overall["regression_mape"] is not None
            and overall["regression_mape"] < gate_target_mape
        ),
        "folds": folds,
    }

    logger.info(
        "Metadata LOO benchmark: %d entries / %d lakes — "
        "Huggel MAPE=%.1f%%, Regression MAPE=%.1f%% (Himalaya: %d entries)",
        result["n_entries_evaluable"],
        result["n_unique_lakes"],
        (overall["huggel_mape"] or 0) * 100,
        (overall["regression_mape"] or 0) * 100,
        len(himalaya),
    )
    return result
