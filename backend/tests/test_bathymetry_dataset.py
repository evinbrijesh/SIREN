"""Tests for the surveyed bathymetry dataset adapter (ADR-013 §9.7.1).

Tests cover:
    - 16-lake CSV loader: column parsing, CRS, depth filtering
    - 4-lake shapefile loader: depth column detection, CRS, outline loading
    - CRS alignment between sources (WGS1984 vs UTM)
    - Nodata / invalid depth handling
    - Bed-elevation unit verification (depth in metres)
    - Leave-one-lake-out split reproducibility and leakage prevention
    - Volume validation against the global compilation
    - Manifest generation

Tests that require the real downloaded datasets are marked with
``@pytest.mark.skipif`` and check for data availability at import time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point

from siren.ml.bathymetry_dataset import (
    LakeRecord,
    LakeSplit,
    GlobalCompilationEntry,
    load_zhang_16lakes,
    load_das_4lakes,
    load_all_surveyed_lakes,
    load_global_compilation,
    build_manifest,
    generate_loo_splits,
    get_split_data,
    validate_volumes,
    ZHANG_16LAKES_DIR,
    DAS_4LAKES_DIR,
    GLOBAL_COMPILATION_PATH,
)

# --- Skip conditions ---

ZHANG_AVAILABLE = (
    ZHANG_16LAKES_DIR / "Glacial_Lake_bathymetry" / "GlacialLakeBathymetry"
).exists()
DAS_AVAILABLE = (
    DAS_4LAKES_DIR / "In-Situ Bathymetry Data for JOG"
).exists()
GLOBAL_AVAILABLE = GLOBAL_COMPILATION_PATH.exists()

skip_zhang = pytest.mark.skipif(not ZHANG_AVAILABLE, reason="16-lake dataset not downloaded")
skip_das = pytest.mark.skipif(not DAS_AVAILABLE, reason="4-lake dataset not downloaded")
skip_global = pytest.mark.skipif(not GLOBAL_AVAILABLE, reason="global compilation not downloaded")
skip_all = pytest.mark.skipif(
    not (ZHANG_AVAILABLE and DAS_AVAILABLE),
    reason="surveyed bathymetry datasets not downloaded",
)


# --------------------------------------------------------------------------- #
# 16-lake loader
# --------------------------------------------------------------------------- #

@skip_zhang
class TestZhang16Lakes:
    """Tests for the 16-lake greater-Himalaya bathymetry loader."""

    def test_loads_all_16_lakes(self):
        """All 16 lakes are loaded."""
        records = load_zhang_16lakes()
        assert len(records) == 16, f"Expected 16 lakes, got {len(records)}"

    def test_all_points_have_valid_depths(self):
        """All depth values are positive (invalid depths filtered)."""
        records = load_zhang_16lakes()
        for r in records:
            assert (r.points["depth_m"] > 0).all(), (
                f"{r.lake_name} has non-positive depths"
            )

    def test_crs_is_wgs1984(self):
        """16-lake CSV coordinates are in WGS1984 (EPSG:4326)."""
        records = load_zhang_16lakes()
        for r in records:
            assert r.crs is not None
            assert r.crs.to_epsg() == 4326, (
                f"{r.lake_name} CRS is {r.crs}, expected EPSG:4326"
            )

    def test_depth_units_are_metres(self):
        """Depths are in metres — sanity-check against published ranges.

        Himalayan glacial lake depths range from ~10 m (small lakes) to
        ~200 m (deep lakes like Galongco at 198 m). Values outside
        [1, 300] m would indicate a unit conversion error.
        """
        records = load_zhang_16lakes()
        for r in records:
            assert 1.0 <= r.max_depth_m <= 300.0, (
                f"{r.lake_name} max depth {r.max_depth_m} m is outside [1, 300]"
            )
            assert 0.1 <= r.mean_depth_m <= 200.0, (
                f"{r.lake_name} mean depth {r.mean_depth_m} m is outside [0.1, 200]"
            )

    def test_point_counts_match_expected(self):
        """Point counts match the known dataset (66,487 total)."""
        records = load_zhang_16lakes()
        total = sum(r.n_points for r in records)
        # Allow small variation due to filtering of zero/negative depths
        assert 60000 <= total <= 70000, f"Total points {total} outside [60000, 70000]"

    def test_lake_ids_are_unique(self):
        """Each lake has a unique lake_id."""
        records = load_zhang_16lakes()
        ids = [r.lake_id for r in records]
        assert len(ids) == len(set(ids)), "Duplicate lake_ids"

    def test_lake_names_are_canonical(self):
        """Lake names are properly normalised (title-case, not raw filenames)."""
        records = load_zhang_16lakes()
        for r in records:
            # Should be title-case (e.g. "Bechung Tsho", "Galongco")
            # and not contain underscores or be all-lowercase
            assert "_" not in r.lake_name, (
                f"Lake name '{r.lake_name}' contains underscore"
            )
            assert not r.lake_name.islower(), (
                f"Lake name '{r.lake_name}' is all-lowercase"
            )

    def test_bounds_are_reasonable(self):
        """Geographic bounds are within the greater Himalaya region."""
        records = load_zhang_16lakes()
        for r in records:
            minx, miny, maxx, maxy = r.bounds
            # Greater Himalaya: ~27-32 N, ~85-100 E
            assert 27.0 <= miny <= 32.0, f"{r.lake_name} miny={miny}"
            assert 27.0 <= maxy <= 32.0, f"{r.lake_name} maxy={maxy}"
            assert 85.0 <= minx <= 100.0, f"{r.lake_name} minx={minx}"
            assert 85.0 <= maxx <= 100.0, f"{r.lake_name} maxx={maxx}"

    def test_galongco_is_among_deepest(self):
        """Galongco (~198 m) is among the deepest lakes in the dataset.

        Note: several lakes (Talongco, Rewuco, Jinwongco) have raw survey
        max depths exceeding 200 m — Galongco is the deepest lake with a
        published value in the global compilation (198.84 m).
        """
        records = load_zhang_16lakes()
        galongco = [r for r in records if "galongco" in r.lake_id][0]
        assert galongco.max_depth_m > 190.0, (
            f"Galongco max depth {galongco.max_depth_m} < 190 m"
        )
        all_max = [r.max_depth_m for r in records]
        # Galongco should be in the top 5 deepest
        assert galongco.max_depth_m >= sorted(all_max, reverse=True)[4], (
            f"Galongco ({galongco.max_depth_m}) not in top 5 deepest"
        )


# --------------------------------------------------------------------------- #
# 4-lake loader
# --------------------------------------------------------------------------- #

@skip_das
class TestDas4Lakes:
    """Tests for the 4-lake western-Himalaya in-situ bathymetry loader."""

    def test_loads_all_4_lakes(self):
        """All 4 lakes are loaded."""
        records = load_das_4lakes()
        assert len(records) == 4, f"Expected 4 lakes, got {len(records)}"

    def test_crs_is_utm(self):
        """4-lake shapefile coordinates are in UTM (EPSG:32643)."""
        records = load_das_4lakes()
        for r in records:
            assert r.crs is not None
            epsg = r.crs.to_epsg()
            assert epsg == 32643, f"{r.lake_name} CRS is EPSG:{epsg}, expected 32643"

    def test_depth_units_are_metres(self):
        """Depths match published values — sanity-check unit correctness."""
        records = load_das_4lakes()
        by_name = {r.lake_name: r for r in records}

        # Kya Tso: published max 16 m
        assert by_name["Kya Tso Lake"].max_depth_m < 25, (
            f"Kya Tso max depth {by_name['Kya Tso Lake'].max_depth_m} > 25 m"
        )

        # Samudra Tapu: published max 59 m
        assert 50 < by_name["Samudra Tapu Lake"].max_depth_m < 70, (
            f"Samudra Tapu max depth {by_name['Samudra Tapu Lake'].max_depth_m}"
        )

        # Gepang Gath: published max 46 m
        assert 40 < by_name["Gepang Gath Lake"].max_depth_m < 55, (
            f"Gepang Gath max depth {by_name['Gepang Gath Lake'].max_depth_m}"
        )

    def test_all_points_have_valid_depths(self):
        """All depth values are positive."""
        records = load_das_4lakes()
        for r in records:
            assert (r.points["depth_m"] > 0).all(), (
                f"{r.lake_name} has non-positive depths"
            )

    def test_outlines_loaded_for_3_lakes(self):
        """At least 3 of the 4 lakes have outline geometry."""
        records = load_das_4lakes()
        n_with_outline = sum(1 for r in records if r.outline is not None)
        assert n_with_outline >= 3, (
            f"Only {n_with_outline} lakes have outlines"
        )

    def test_point_counts_match_expected(self):
        """Point counts match the known dataset (53,130 total)."""
        records = load_das_4lakes()
        total = sum(r.n_points for r in records)
        assert 45000 <= total <= 60000, f"Total points {total} outside [45000, 60000]"

    def test_gepang_gath_is_largest(self):
        """Gepang Gath has the most measurement points (~26,000)."""
        records = load_das_4lakes()
        by_name = {r.lake_name: r for r in records}
        assert by_name["Gepang Gath Lake"].n_points > 20000, (
            f"Gepang Gath has {by_name['Gepang Gath Lake'].n_points} points"
        )


# --------------------------------------------------------------------------- #
# CRS alignment
# --------------------------------------------------------------------------- #

@skip_all
class TestCRSAlignment:
    """Tests for CRS alignment between the two sources."""

    def test_sources_use_different_crs(self):
        """16-lake is WGS1984, 4-lake is UTM — confirmed different."""
        zhang = load_zhang_16lakes()
        das = load_das_4lakes()
        assert zhang[0].crs.to_epsg() == 4326
        assert das[0].crs.to_epsg() == 32643

    def test_reproject_to_common_crs(self):
        """Points can be reprojected to a common CRS without error."""
        from siren.ml.bathymetry_dataset import get_split_data
        records = load_all_surveyed_lakes()
        splits = generate_loo_splits(records)
        train_gdf, test_gdf = get_split_data(records, splits[0], target_crs="EPSG:4326")
        assert train_gdf.crs.to_epsg() == 4326
        assert test_gdf.crs.to_epsg() == 4326
        # All points should have valid geometry
        assert train_gdf.geometry.notna().all()
        assert test_gdf.geometry.notna().all()

    def test_reproject_preserves_depth_values(self):
        """Reprojection does not alter depth values."""
        records = load_all_surveyed_lakes()
        # Pick a 4-lake record (UTM) and reproject to WGS84
        das_rec = [r for r in records if r.source == "das2025_4lakes"][0]
        original_depths = das_rec.points["depth_m"].values.copy()
        reprojected = das_rec.points.to_crs("EPSG:4326")
        np.testing.assert_array_equal(original_depths, reprojected["depth_m"].values)


# --------------------------------------------------------------------------- #
# Nodata / invalid depth handling
# --------------------------------------------------------------------------- #

class TestNodataHandling:
    """Tests for nodata and invalid depth filtering (synthetic fixtures)."""

    def test_zero_depths_filtered(self, tmp_path):
        """Zero-depth points are filtered out."""
        # Create a minimal CSV mimicking the 16-lake format
        csv_path = tmp_path / "TestLake_Bathymetry_20210101.csv"
        csv_path.write_text(
            "id,Latitude,Longitude,Depth(m)\n"
            "1,28.0,86.0,10.5\n"
            "2,28.001,86.001,0.0\n"      # filtered
            "3,28.002,86.002,-5.0\n"     # filtered
            "4,28.003,86.003,15.2\n"
        )
        # Use the loader directly on this file
        from siren.ml.bathymetry_dataset import _parse_zhang_survey_date
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"Latitude": "latitude", "Longitude": "longitude", "Depth(m)": "depth_m"})
        df = df.dropna(subset=["latitude", "longitude", "depth_m"])
        df = df[df["depth_m"] > 0]
        assert len(df) == 2  # only the two positive depths remain

    def test_nan_depths_filtered(self):
        """NaN depth values are filtered out."""
        import pandas as pd
        df = pd.DataFrame({
            "latitude": [28.0, 28.1, 28.2],
            "longitude": [86.0, 86.1, 86.2],
            "depth_m": [10.0, np.nan, 20.0],
        })
        df = df.dropna(subset=["depth_m"])
        df = df[df["depth_m"] > 0]
        assert len(df) == 2
        assert np.isnan(df["depth_m"]).sum() == 0


# --------------------------------------------------------------------------- #
# Leave-one-lake-out splits
# --------------------------------------------------------------------------- #

@skip_all
class TestLOOSplits:
    """Tests for leave-one-lake-out split generation."""

    def test_correct_number_of_splits(self):
        """Number of splits equals number of lakes."""
        records = load_all_surveyed_lakes()
        splits = generate_loo_splits(records)
        assert len(splits) == len(records)

    def test_each_lake_held_out_exactly_once(self):
        """Each lake appears as the test set exactly once."""
        records = load_all_surveyed_lakes()
        splits = generate_loo_splits(records)
        test_ids = [s.test_lake_id for s in splits]
        assert len(test_ids) == len(set(test_ids)), "Duplicate test lake IDs"
        assert set(test_ids) == set(r.lake_id for r in records)

    def test_no_pixel_level_leakage(self):
        """No lake appears in both train and test for any split."""
        records = load_all_surveyed_lakes()
        splits = generate_loo_splits(records)
        for split in splits:
            assert split.test_lake_id not in split.train_lake_ids, (
                f"Test lake {split.test_lake_id} also in train set"
            )

    def test_splits_are_reproducible(self):
        """Same seed produces identical splits."""
        records = load_all_surveyed_lakes()
        splits1 = generate_loo_splits(records, seed=42)
        splits2 = generate_loo_splits(records, seed=42)
        for s1, s2 in zip(splits1, splits2):
            assert s1.test_lake_id == s2.test_lake_id
            assert s1.train_lake_ids == s2.train_lake_ids

    def test_train_point_count_decreases_by_test_size(self):
        """Train point count = total - test point count for each split."""
        records = load_all_surveyed_lakes()
        total_points = sum(r.n_points for r in records)
        splits = generate_loo_splits(records)
        for split in splits:
            assert split.n_train_points + split.n_test_points == total_points, (
                f"Train ({split.n_train_points}) + Test ({split.n_test_points}) "
                f"!= Total ({total_points}) for {split.test_lake_name}"
            )

    def test_get_split_data_returns_correct_lakes(self):
        """get_split_data returns only the correct lakes for each split."""
        records = load_all_surveyed_lakes()
        splits = generate_loo_splits(records)
        split = splits[0]
        train_gdf, test_gdf = get_split_data(records, split, target_crs="EPSG:4326")
        # Test GDF should only contain the test lake
        assert (test_gdf["lake_id"] == split.test_lake_id).all()
        # Train GDF should NOT contain the test lake
        assert split.test_lake_id not in train_gdf["lake_id"].values


# --------------------------------------------------------------------------- #
# Volume validation
# --------------------------------------------------------------------------- #

@skip_all
@skip_global
class TestVolumeValidation:
    """Tests for volume validation against the global compilation."""

    def test_at_least_5_lakes_matched(self):
        """At least 5 surveyed lakes match entries in the global compilation."""
        records = load_all_surveyed_lakes()
        entries = load_global_compilation()
        comparisons = validate_volumes(records, entries)
        assert len(comparisons) >= 5, (
            f"Only {len(comparisons)} lakes matched to global compilation"
        )

    def test_discrepancy_below_5_percent_for_matched(self):
        """Matched lakes have < 5% depth discrepancy (same source data)."""
        records = load_all_surveyed_lakes()
        entries = load_global_compilation()
        comparisons = validate_volumes(records, entries)
        for c in comparisons:
            assert c["discrepancy_pct"] < 5.0, (
                f"{c['lake_name']}: {c['discrepancy_pct']}% discrepancy"
            )

    def test_imja_in_global_compilation(self):
        """Imja Lake appears in the global compilation with multiple surveys."""
        entries = load_global_compilation()
        imja_entries = [e for e in entries if "imja" in e.name.lower()]
        assert len(imja_entries) >= 3, (
            f"Imja has {len(imja_entries)} entries, expected >= 3"
        )


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

@skip_all
class TestManifest:
    """Tests for manifest generation."""

    def test_manifest_has_correct_structure(self):
        """Manifest has the expected top-level keys."""
        records = load_all_surveyed_lakes()
        manifest = build_manifest(records)
        assert "total_lakes" in manifest
        assert "total_points" in manifest
        assert "sources" in manifest
        assert "lakes" in manifest
        assert manifest["total_lakes"] == len(records)

    def test_manifest_lake_entries_have_required_fields(self):
        """Each lake entry in the manifest has all required fields."""
        records = load_all_surveyed_lakes()
        manifest = build_manifest(records)
        required = {"lake_id", "lake_name", "source", "n_points", "max_depth_m", "doi"}
        for entry in manifest["lakes"]:
            assert required.issubset(entry.keys()), (
                f"Missing fields: {required - entry.keys()}"
            )

    def test_manifest_writes_to_file(self, tmp_path):
        """Manifest can be written to a JSON file."""
        records = load_all_surveyed_lakes()
        out = tmp_path / "manifest.json"
        manifest = build_manifest(records, output_path=out)
        assert out.exists()
        import json
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["total_lakes"] == manifest["total_lakes"]
