"""Surveyed bathymetry dataset adapter (ADR-013 §9.7.1 — E2 real data path).

Unified loader for surveyed glacial-lake bathymetry from two independent
field-survey sources, plus a metadata-level global compilation for volume
benchmarking. This replaces the earlier reliance on Millan/Farinotti glacier
ice-thickness estimates, which are proxy data — not lake-bed ground truth.

Sources loaded by this module:
    1. Greater-Himalaya 16-lake bathymetry (Zhang et al. 2023, Figshare)
       — per-lake CSV measurement points (id, Latitude, Longitude, Depth(m))
         in WGS1984, plus Kriging interpolation grids (.grd) and shapefiles.
    2. Western-Himalaya 4-lake in-situ bathymetry (Das & Ramsankaran 2025,
       Zenodo) — per-lake shapefiles with 24-column schema including
       EchoDepth, CRS EPSG:32643. Includes historical lake outlines.
    3. Global glacial-lake bathymetry compilation (Zhang & Wang 2023, Zenodo)
       — metadata-only XLSX with lake name, location, survey year, area,
         volume, max depth. Used for volume validation, not dense bed grids.

The adapter normalises all sources to a common schema:

    LakeRecord:
        lake_id        — stable identifier (source-prefixed)
        lake_name      — human-readable name
        source         — "zhang2023_16lakes" | "das2025_4lakes"
        survey_date    — date or date range string
        crs            — original CRS of the measurements
        points         — GeoDataFrame with columns: geometry (Point),
                         depth_m (float), survey_source
        n_points       — number of measurement points
        bounds         — (minx, miny, maxx, maxy) in the original CRS
        max_depth_m    — maximum surveyed depth
        mean_depth_m   — mean surveyed depth
        outline        — lake outline geometry (if available, else None)

Leave-one-lake-out splits:
    The split generator produces K folds where each fold holds out exactly
    one lake as the test set and uses the remaining lakes for training.
    This prevents pixel-level leakage — neighbouring depth points from the
    same lake never appear in both train and test.

Provenance:
    Every LakeRecord carries its source, DOI, and license. The manifest
    builder writes a JSON sidecar recording all lakes, their metadata, and
    the split assignment.

Scientific notes:
    - Surveyed bathymetry is real ground truth, distinct from glacier
      ice-thickness proxies (Millan/Farinotti).
    - The 16-lake and 4-lake datasets use different survey methods
      (echo sounder vs USV-mounted echo sounder) and different CRS — the
      adapter handles CRS normalisation internally.
    - Lake outlines from the 4-lake dataset are time-series (22 years for
      Samudra Tapu) — the adapter selects the most recent outline by
      default but exposes the full time-series for expansion analysis.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# --- Dataset paths (relative to repo root) ---

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO_ROOT / "data" / "datasets"

ZHANG_16LAKES_DIR = DATASETS_DIR / "bathymetry_himalaya_16lakes"
DAS_4LAKES_DIR = DATASETS_DIR / "bathymetry_western_himalaya_4lakes"
GLOBAL_COMPILATION_PATH = (
    DATASETS_DIR
    / "bathymetry_global_compilation"
    / "Global_bathymetric_survey_for_glacial_lakes.xlsx"
)

# --- Provenance constants ---

ZHANG_16LAKES_DOI = "10.6084/m9.figshare.21569175.v1"
ZHANG_16LAKES_LICENSE = "CC BY 4.0"
ZHANG_16LAKES_CITATION = (
    "Zhang, G., T. Bolch, T. Yao, D. R. Rounce, W. Chen, G. Veh, O. King, "
    "S. K. Allen, M. Wang, and W. Wang (2023), Underestimated mass loss "
    "from lake-terminating glaciers in the greater Himalaya, Nature "
    "Geoscience, doi: 10.1038/s41561-023-01150-1"
)

DAS_4LAKES_DOI = "10.5281/zenodo.16677320"
DAS_4LAKES_LICENSE = "CC BY 4.0"
DAS_4LAKES_CITATION = (
    "Das, S. and Ramsankaran, R. (2025), In-situ bathymetry and volume "
    "estimation of four glacial lakes in western Himalaya, Journal of "
    "Glaciology"
)

GLOBAL_COMPILATION_DOI = "10.5281/zenodo.10201073"
GLOBAL_COMPILATION_LICENSE = "CC BY 4.0"

# --- Lake name normalisation ---

# The 16-lake dataset uses inconsistent casing in filenames. Map the
# filename prefix to a canonical lake name.
_ZHANG_LAKE_NAMES: dict[str, str] = {
    "bechungtsho": "Bechung Tsho",
    "bencoguoco": "Bencoguoco",
    "bielongco": "Bielongco",
    "cirenmaco": "Cirenmaco",
    "galongco": "Galongco",
    "guangxieco": "Guangxieco",
    "jialongco": "Jialongco",
    "jinwongco": "Jinwongco",
    "luggye": "Luggye Tsho",
    "maqiongco": "Maqiongco",
    "poiquno.1": "Poiqu No.1",
    "ranzeriaco": "Ranzeriaco",
    "raphstreng": "Raphstreng Tsho",
    "rewuco": "Rewuco",
    "shishapangmano.1": "Shishapangma No.1",
    "talongco": "Talongco",
}

# The 4-lake dataset directory names map to canonical names.
_DAS_LAKE_NAMES: dict[str, str] = {
    "01-kya-tso-lake": "Kya Tso Lake",
    "02-panchi-nala-lake": "Panchi Nala Lake",
    "03-gepang-gath-lake": "Gepang Gath Lake",
    "04-samudri-tapu-lake": "Samudra Tapu Lake",
}

# Published max depths and volumes for the 4-lake dataset (from the Zenodo
# record description). Used for sanity-checking loaded data.
_DAS_PUBLISHED = {
    "Kya Tso Lake": {"max_depth_m": 16, "volume_mcm": 0.89},
    "Panchi Nala Lake": {"max_depth_m": 10, "volume_mcm": 0.44},
    "Gepang Gath Lake": {"max_depth_m": 46, "volume_mcm": 24.12},
    "Samudra Tapu Lake": {"max_depth_m": 59, "volume_mcm": 24.69},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LakeRecord:
    """A single surveyed lake with its depth measurement points.

    Attributes:
        lake_id: stable identifier prefixed by source (e.g. "zhang16:galongco").
        lake_name: canonical human-readable name.
        source: dataset source tag ("zhang2023_16lakes" or "das2025_4lakes").
        survey_date: date or date range string from the source.
        crs: original CRS of the measurement points (pyproj CRS object).
        points: GeoDataFrame with columns [geometry, depth_m, survey_source].
        n_points: number of measurement points.
        bounds: (minx, miny, maxx, maxy) in the original CRS.
        max_depth_m: maximum surveyed depth in metres.
        mean_depth_m: mean surveyed depth in metres.
        outline: lake outline geometry (most recent, if available), else None.
        doi: DOI of the source dataset.
        license: license of the source dataset.
        provenance: full provenance dict for JSON serialisation.
    """

    lake_id: str
    lake_name: str
    source: str
    survey_date: str
    crs: Any
    points: gpd.GeoDataFrame
    n_points: int
    bounds: tuple[float, float, float, float]
    max_depth_m: float
    mean_depth_m: float
    outline: Any | None = None
    doi: str = ""
    license: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_manifest_entry(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for the manifest."""
        return {
            "lake_id": self.lake_id,
            "lake_name": self.lake_name,
            "source": self.source,
            "survey_date": self.survey_date,
            "crs": str(self.crs),
            "n_points": self.n_points,
            "bounds": list(self.bounds),
            "max_depth_m": round(self.max_depth_m, 2),
            "mean_depth_m": round(self.mean_depth_m, 2),
            "has_outline": self.outline is not None,
            "doi": self.doi,
            "license": self.license,
        }


@dataclass
class GlobalCompilationEntry:
    """A single entry from the global bathymetry compilation (metadata only).

    Attributes:
        name: lake name.
        mountain: mountain range.
        country: country.
        lon: longitude (DMS string from the source).
        lat: latitude (DMS string from the source).
        survey_year: year of survey (int or None).
        area_km2: lake surface area in km².
        volume_mcm: lake volume in millions of m³.
        max_depth_m: maximum depth in metres (or None).
        source: source publication string.
        lake_type: worksheet the entry came from — one of "proglacial",
            "periglacial", "extraglacial", "supraglacial", "ice-dammed".
    """

    name: str
    mountain: str
    country: str
    lon: str
    lat: str
    survey_year: int | None
    area_km2: float | None
    volume_mcm: float | None
    max_depth_m: float | None
    source: str
    lake_type: str = ""


# ---------------------------------------------------------------------------
# 16-lake greater-Himalaya loader (Zhang et al. 2023)
# ---------------------------------------------------------------------------


def _parse_zhang_survey_date(filename: str) -> str:
    """Extract the survey date from a 16-lake CSV filename.

    Filenames look like ``BechungTsho_Bathymetry_20211011.csv`` or
    ``Galongco_Bathymetry_20191002_20201015.csv`` (two dates = two surveys).
    """
    # Match all 8-digit date groups in the filename
    dates = re.findall(r"(\d{8})", filename)
    if not dates:
        return "unknown"
    # Format as YYYY-MM-DD
    formatted = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates]
    return " to ".join(formatted)


def load_zhang_16lakes(
    data_dir: Path | None = None,
) -> list[LakeRecord]:
    """Load the 16-lake greater-Himalaya bathymetry dataset.

    Each lake has a CSV with columns (id, Latitude, Longitude, Depth(m)) in
    WGS1984. The loader normalises the lake name from the filename prefix
    and builds a GeoDataFrame of measurement points.

    Args:
        data_dir: path to the extracted dataset directory. Defaults to
            ``data/datasets/bathymetry_himalaya_16lakes``.

    Returns:
        List of LakeRecord objects, one per lake.
    """
    if data_dir is None:
        data_dir = ZHANG_16LAKES_DIR
    bathy_dir = data_dir / "Glacial_Lake_bathymetry" / "GlacialLakeBathymetry"
    if not bathy_dir.exists():
        logger.warning("16-lake bathymetry directory not found: %s", bathy_dir)
        return []

    records: list[LakeRecord] = []
    csv_files = sorted(bathy_dir.glob("*_Bathymetry_*.csv"))
    # Also match files without "Bathymetry" in the name:
    #   Rewuco_20211025.csv, ShishapangmaNo.1_20210920.csv
    csv_files += sorted(bathy_dir.glob("Rewuco_*.csv"))
    csv_files += sorted(bathy_dir.glob("ShishapangmaNo.1_*.csv"))
    csv_files = sorted(set(csv_files))

    for csv_path in csv_files:
        # Extract lake name from filename prefix
        stem = csv_path.stem.lower()
        # Remove the _Bathymetry_... or _date suffix
        for prefix, canonical in _ZHANG_LAKE_NAMES.items():
            if stem.startswith(prefix):
                lake_name = canonical
                break
        else:
            # Fallback: use the part before the first _Bathymetry or _date
            parts = re.split(r"_bathymetry_|_\d{8}", stem)
            lake_name = parts[0].replace("_", " ").title()

        survey_date = _parse_zhang_survey_date(csv_path.name)

        # Read CSV
        df = pd.read_csv(csv_path)
        # Expected columns: id, Latitude, Longitude, Depth(m)
        # Normalise column names (handle case variations)
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "lat" in cl:
                col_map[c] = "latitude"
            elif "lon" in cl:
                col_map[c] = "longitude"
            elif "depth" in cl:
                col_map[c] = "depth_m"
            elif cl == "id":
                col_map[c] = "id"
        df = df.rename(columns=col_map)

        if "depth_m" not in df.columns or "latitude" not in df.columns:
            logger.warning("Skipping %s: missing required columns", csv_path.name)
            continue

        # Drop rows with NaN depth or coordinates
        df = df.dropna(subset=["latitude", "longitude", "depth_m"])
        # Filter out zero or negative depths (measurement noise)
        df = df[df["depth_m"] > 0]
        # Filter out zero coordinates (missing GPS — some Luggye points have lon=0)
        df = df[(df["longitude"] != 0) & (df["latitude"] != 0)]
        # Filter out coordinates outside the greater Himalaya range
        # (data entry errors — some Luggye points have lon=0.5, lat=81.9)
        df = df[
            (df["longitude"] > 80) & (df["longitude"] < 105)
            & (df["latitude"] > 25) & (df["latitude"] < 36)
        ]

        if df.empty:
            logger.warning("Skipping %s: no valid depth points after filtering", csv_path.name)
            continue

        # Build GeoDataFrame in WGS1984
        geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(
            df[["depth_m"]].copy(),
            geometry=geometry,
            crs="EPSG:4326",
        )
        gdf["survey_source"] = "zhang2023_16lakes"

        lake_id = f"zhang16:{lake_name.lower().replace(' ', '_')}"
        bounds = tuple(gdf.total_bounds)  # (minx, miny, maxx, maxy)

        record = LakeRecord(
            lake_id=lake_id,
            lake_name=lake_name,
            source="zhang2023_16lakes",
            survey_date=survey_date,
            crs=gdf.crs,
            points=gdf,
            n_points=len(gdf),
            bounds=bounds,
            max_depth_m=float(gdf["depth_m"].max()),
            mean_depth_m=float(gdf["depth_m"].mean()),
            doi=ZHANG_16LAKES_DOI,
            license=ZHANG_16LAKES_LICENSE,
            provenance={
                "source": "Zhang et al. 2023 (Figshare 21569175)",
                "doi": ZHANG_16LAKES_DOI,
                "license": ZHANG_16LAKES_LICENSE,
                "citation": ZHANG_16LAKES_CITATION,
                "survey_method": "echo sounder",
                "coordinate_system": "WGS1984 (EPSG:4326)",
            },
        )
        records.append(record)
        logger.debug(
            "Loaded %s: %d points, max_depth=%.1f m",
            lake_name,
            record.n_points,
            record.max_depth_m,
        )

    logger.info("Loaded %d lakes from Zhang 16-lake dataset", len(records))
    return records


# ---------------------------------------------------------------------------
# 4-lake western-Himalaya loader (Das & Ramsankaran 2025)
# ---------------------------------------------------------------------------


def _find_depth_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Find the depth column in a 4-lake shapefile GeoDataFrame.

    The 4-lake dataset has inconsistent schemas:
        - Kya Tso / Gepang Gath / Samudra Tapu: 'EchoDepth(' column
        - Panchi Nala: 'Depth(m)' column
    """
    for col in gdf.columns:
        cl = col.lower().strip()
        if cl in ("echodepth(", "echodepth", "depth(m)", "depth"):
            return col
        if cl.startswith("echodepth") or cl.startswith("depth("):
            return col
    # Last resort: any column containing 'depth' (case-insensitive)
    depth_cols = [c for c in gdf.columns if "depth" in c.lower()]
    if depth_cols:
        return depth_cols[0]
    return None


def _load_latest_outline(outline_dir: Path) -> Any | None:
    """Load the most recent lake outline shapefile from a directory.

    The 4-lake dataset includes time-series outlines named by date
    (e.g. ``2024-08-22.shp``). This function picks the latest one.
    """
    shp_files = sorted(outline_dir.glob("*.shp"))
    if not shp_files:
        return None
    # Sort by filename (dates sort lexicographically)
    latest = shp_files[-1]
    try:
        gdf = gpd.read_file(latest)
        if len(gdf) > 0:
            return gdf.geometry.iloc[0]
    except Exception as e:
        logger.warning("Failed to load outline %s: %s", latest.name, e)
    return None


def load_das_4lakes(
    data_dir: Path | None = None,
) -> list[LakeRecord]:
    """Load the 4-lake western-Himalaya in-situ bathymetry dataset.

    Each lake has a shapefile with depth measurement points (CRS EPSG:32643
    or similar UTM zone). The loader finds the depth column, builds a
    GeoDataFrame, and loads the most recent lake outline if available.

    Args:
        data_dir: path to the extracted dataset directory. Defaults to
            ``data/datasets/bathymetry_western_himalaya_4lakes``.

    Returns:
        List of LakeRecord objects, one per lake.
    """
    if data_dir is None:
        data_dir = DAS_4LAKES_DIR
    bathy_base = data_dir / "In-Situ Bathymetry Data for JOG"
    if not bathy_base.exists():
        logger.warning("4-lake bathymetry directory not found: %s", bathy_base)
        return []

    bathy_dir = bathy_base / "2 Bathymetry data"
    outline_base = bathy_base / "3 Lake outlines"

    records: list[LakeRecord] = []

    for lake_dir in sorted(bathy_dir.iterdir()):
        if not lake_dir.is_dir():
            continue
        dir_name = lake_dir.name.lower()
        lake_name = _DAS_LAKE_NAMES.get(dir_name, lake_dir.name)

        # Find the depth shapefile
        shp_files = sorted(lake_dir.glob("*.shp"))
        if not shp_files:
            logger.warning("No shapefile in %s, skipping", lake_dir.name)
            continue

        gdf = gpd.read_file(shp_files[0])
        depth_col = _find_depth_column(gdf)
        if depth_col is None:
            logger.warning("No depth column found in %s, skipping", shp_files[0].name)
            continue

        # Extract depth values
        depths = pd.to_numeric(gdf[depth_col], errors="coerce")
        gdf = gdf.copy()
        gdf["depth_m"] = depths
        gdf = gdf.dropna(subset=["depth_m"])
        gdf = gdf[gdf["depth_m"] > 0]
        gdf["survey_source"] = "das2025_4lakes"

        if gdf.empty:
            logger.warning("No valid depth points in %s, skipping", shp_files[0].name)
            continue

        # Keep only geometry and depth_m
        keep_cols = ["depth_m", "survey_source", "geometry"]
        gdf = gdf[keep_cols].reset_index(drop=True)

        # Determine survey date from the directory or filename
        survey_date = "2022-08 to 2024-08"
        if "14sept2022" in shp_files[0].name.lower():
            survey_date = "2022-09-14"

        # Load outline if available
        outline = None
        outline_dir = outline_base / lake_name
        if outline_dir.exists():
            outline = _load_latest_outline(outline_dir)

        lake_id = f"das4:{lake_name.lower().replace(' ', '_')}"
        bounds = tuple(gdf.total_bounds)

        # Sanity-check against published values
        published = _DAS_PUBLISHED.get(lake_name, {})
        max_depth = float(gdf["depth_m"].max())
        if published and "max_depth_m" in published:
            expected = published["max_depth_m"]
            # Allow 20% tolerance — survey methods may differ slightly
            if max_depth > expected * 1.5:
                logger.warning(
                    "%s: max depth %.1f m exceeds published %.1f m by >50%% — "
                    "check depth column selection",
                    lake_name,
                    max_depth,
                    expected,
                )

        record = LakeRecord(
            lake_id=lake_id,
            lake_name=lake_name,
            source="das2025_4lakes",
            survey_date=survey_date,
            crs=gdf.crs,
            points=gdf,
            n_points=len(gdf),
            bounds=bounds,
            max_depth_m=max_depth,
            mean_depth_m=float(gdf["depth_m"].mean()),
            outline=outline,
            doi=DAS_4LAKES_DOI,
            license=DAS_4LAKES_LICENSE,
            provenance={
                "source": "Das & Ramsankaran 2025 (Zenodo 16677320)",
                "doi": DAS_4LAKES_DOI,
                "license": DAS_4LAKES_LICENSE,
                "citation": DAS_4LAKES_CITATION,
                "survey_method": "USV-mounted echo sounder",
                "coordinate_system": str(gdf.crs),
            },
        )
        records.append(record)
        logger.debug(
            "Loaded %s: %d points, max_depth=%.1f m, crs=%s",
            lake_name,
            record.n_points,
            record.max_depth_m,
            record.crs,
        )

    logger.info("Loaded %d lakes from Das 4-lake dataset", len(records))
    return records


# ---------------------------------------------------------------------------
# Global compilation loader (metadata only)
# ---------------------------------------------------------------------------


def _parse_survey_year(value: Any) -> int | None:
    """Parse a survey year that may be an int, float, or annotated string
    like '<2014' or 'c.2009'. Returns the first 4-digit year found."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    m = re.search(r"(\d{4})", str(value))
    return int(m.group(1)) if m else None


def load_global_compilation(
    path: Path | None = None,
) -> list[GlobalCompilationEntry]:
    """Load the global glacial-lake bathymetry compilation (metadata only).

    This is a spreadsheet with 323 entries across five worksheets
    (proglacial, periglacial, extraglacial, supraglacial, ice-dammed)
    covering ~250 unique lakes. It provides published area, volume, and
    max depth for benchmarking — not dense bed-elevation ground truth.

    Args:
        path: path to the XLSX file. Defaults to the standard location.

    Returns:
        List of GlobalCompilationEntry objects.
    """
    if path is None:
        path = GLOBAL_COMPILATION_PATH
    if not path.exists():
        logger.warning("Global compilation not found: %s", path)
        return []

    sheets = pd.read_excel(path, sheet_name=None)
    entries: list[GlobalCompilationEntry] = []
    for lake_type, df in sheets.items():
        for _, row in df.iterrows():
            entries.append(
                GlobalCompilationEntry(
                    name=str(row.get("Name", "")),
                    mountain=str(row.get("Mountain", "")),
                    country=str(row.get("Country", "")),
                    lon=str(row.get("Lon", "")),
                    lat=str(row.get("Lat", "")),
                    survey_year=_parse_survey_year(row.get("Survey time")),
                    area_km2=float(row["Area/km2"]) if pd.notna(row.get("Area/km2")) else None,
                    volume_mcm=float(row["Volume/m6"]) if pd.notna(row.get("Volume/m6")) else None,
                    max_depth_m=float(row["Max Depth"]) if pd.notna(row.get("Max Depth")) else None,
                    source=str(row.get("Source", "")),
                    lake_type=str(lake_type),
                )
            )
    logger.info("Loaded %d entries from global compilation", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Unified loader + manifest
# ---------------------------------------------------------------------------


def load_all_surveyed_lakes(
    zhang_dir: Path | None = None,
    das_dir: Path | None = None,
) -> list[LakeRecord]:
    """Load all surveyed bathymetry lakes from both sources.

    Returns a single list of LakeRecord objects from both the 16-lake
    and 4-lake datasets. CRS is preserved per-lake (callers must
    reproject as needed).

    Args:
        zhang_dir: override path for the 16-lake dataset.
        das_dir: override path for the 4-lake dataset.

    Returns:
        Combined list of LakeRecord objects.
    """
    records: list[LakeRecord] = []
    records.extend(load_zhang_16lakes(zhang_dir))
    records.extend(load_das_4lakes(das_dir))
    logger.info("Total surveyed lakes loaded: %d", len(records))
    return records


def build_manifest(
    records: list[LakeRecord],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe manifest of all surveyed lakes.

    The manifest records per-lake metadata (name, source, CRS, point count,
    depth statistics, bounds, outline availability) and provenance (DOI,
    license). It is suitable for committing to the repository as a
    human-readable dataset inventory.

    Args:
        records: list of LakeRecord objects.
        output_path: if given, write the manifest JSON to this path.

    Returns:
        The manifest as a dict.
    """
    manifest: dict[str, Any] = {
        "description": "Surveyed glacial-lake bathymetry dataset manifest",
        "total_lakes": len(records),
        "total_points": sum(r.n_points for r in records),
        "sources": {
            "zhang2023_16lakes": {
                "doi": ZHANG_16LAKES_DOI,
                "license": ZHANG_16LAKES_LICENSE,
                "citation": ZHANG_16LAKES_CITATION,
                "n_lakes": sum(1 for r in records if r.source == "zhang2023_16lakes"),
            },
            "das2025_4lakes": {
                "doi": DAS_4LAKES_DOI,
                "license": DAS_4LAKES_LICENSE,
                "citation": DAS_4LAKES_CITATION,
                "n_lakes": sum(1 for r in records if r.source == "das2025_4lakes"),
            },
        },
        "lakes": [r.to_manifest_entry() for r in sorted(records, key=lambda r: r.lake_id)],
    }

    if output_path is not None:
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Manifest written: %s", output_path)

    return manifest


# ---------------------------------------------------------------------------
# Leave-one-lake-out split generator
# ---------------------------------------------------------------------------


@dataclass
class LakeSplit:
    """A single leave-one-lake-out split.

    Attributes:
        test_lake_id: the lake held out for testing.
        test_lake_name: human-readable name of the test lake.
        train_lake_ids: list of lake IDs used for training.
        n_train_points: total training points across all train lakes.
        n_test_points: total test points in the held-out lake.
    """

    test_lake_id: str
    test_lake_name: str
    train_lake_ids: list[str]
    n_train_points: int
    n_test_points: int


def generate_loo_splits(
    records: list[LakeRecord],
    seed: int = 42,
) -> list[LakeSplit]:
    """Generate leave-one-lake-out splits.

    Each split holds out exactly one lake as the test set and uses the
    remaining lakes for training. This prevents pixel-level leakage:
    no depth points from the same lake appear in both train and test.

    The splits are deterministic (seeded) and the order of splits is
    sorted by lake_id for reproducibility.

    Args:
        records: list of LakeRecord objects.
        seed: random seed for any internal shuffling (currently unused
            but reserved for future stratified subsampling).

    Returns:
        List of LakeSplit objects, one per lake.
    """
    rng = np.random.default_rng(seed)
    sorted_records = sorted(records, key=lambda r: r.lake_id)

    splits: list[LakeSplit] = []
    for i, test_rec in enumerate(sorted_records):
        train_recs = [r for j, r in enumerate(sorted_records) if j != i]
        split = LakeSplit(
            test_lake_id=test_rec.lake_id,
            test_lake_name=test_rec.lake_name,
            train_lake_ids=[r.lake_id for r in train_recs],
            n_train_points=sum(r.n_points for r in train_recs),
            n_test_points=test_rec.n_points,
        )
        splits.append(split)

    logger.info(
        "Generated %d leave-one-lake-out splits (total: %d train points, %d test points per fold avg)",
        len(splits),
        splits[0].n_train_points if splits else 0,
        np.mean([s.n_test_points for s in splits]) if splits else 0,
    )
    return splits


def get_split_data(
    records: list[LakeRecord],
    split: LakeSplit,
    target_crs: str = "EPSG:4326",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Get the train and test GeoDataFrames for a given split.

    Reprojects all points to a common CRS for spatial operations.

    Args:
        records: list of all LakeRecord objects.
        split: the LakeSplit to materialise.
        target_crs: CRS to reproject all points to.

    Returns:
        (train_gdf, test_gdf) — GeoDataFrames with columns
        [depth_m, survey_source, lake_id, geometry] in the target CRS.
    """
    rec_by_id = {r.lake_id: r for r in records}
    train_parts: list[gpd.GeoDataFrame] = []
    for lid in split.train_lake_ids:
        rec = rec_by_id[lid]
        gdf = rec.points.copy()
        gdf["lake_id"] = lid
        train_parts.append(gdf.to_crs(target_crs))

    test_rec = rec_by_id[split.test_lake_id]
    test_gdf = test_rec.points.copy()
    test_gdf["lake_id"] = split.test_lake_id
    test_gdf = test_gdf.to_crs(target_crs)

    train_gdf = gpd.GeoDataFrame(pd.concat(train_parts, ignore_index=True), crs=target_crs)
    return train_gdf, test_gdf


# ---------------------------------------------------------------------------
# Volume validation against the global compilation
# ---------------------------------------------------------------------------


def validate_volumes(
    records: list[LakeRecord],
    global_entries: list[GlobalCompilationEntry] | None = None,
    lake_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Cross-check surveyed lake volumes against the global compilation.

    For lakes that appear in both the surveyed datasets and the global
    compilation, compare the surveyed max depth against the published
    value. This is a sanity check, not a gate — survey methods differ.

    Args:
        records: surveyed LakeRecord objects.
        global_entries: global compilation entries (loaded if None).
        lake_types: if given, only compare against entries from these
            worksheets. The dense surveys are proglacial moraine-dammed
            lakes; names also appear in other sheets with incomparable
            published values (e.g. Bencoguoco's periglacial entry), so
            pass {"proglacial"} to restrict the comparison.

    Returns:
        List of comparison dicts with lake name, surveyed max depth,
        published max depth, and discrepancy percentage.
    """
    if global_entries is None:
        global_entries = load_global_compilation()
    if lake_types is not None:
        global_entries = [e for e in global_entries if e.lake_type in lake_types]

    # Build a lookup by normalised name
    global_by_name: dict[str, GlobalCompilationEntry] = {}
    for entry in global_entries:
        key = entry.name.lower().strip()
        global_by_name[key] = entry

    comparisons: list[dict[str, Any]] = []
    for rec in records:
        # Try to match by name
        name_key = rec.lake_name.lower().strip()
        # Also try common variations
        match = global_by_name.get(name_key)
        if match is None:
            # Try without "tsho" / "co" suffixes
            for suffix in [" tsho", " co", " lake"]:
                stripped = name_key.replace(suffix, "")
                match = global_by_name.get(stripped)
                if match:
                    break
        if match is None:
            continue

        published_depth = match.max_depth_m
        if published_depth is None:
            continue

        surveyed_depth = rec.max_depth_m
        discrepancy_pct = abs(surveyed_depth - published_depth) / published_depth * 100

        comparisons.append({
            "lake_name": rec.lake_name,
            "lake_id": rec.lake_id,
            "surveyed_max_depth_m": round(surveyed_depth, 2),
            "published_max_depth_m": published_depth,
            "discrepancy_pct": round(discrepancy_pct, 1),
            "published_volume_mcm": match.volume_mcm,
            "published_area_km2": match.area_km2,
            "published_survey_year": match.survey_year,
        })

    logger.info("Volume validation: %d lakes matched to global compilation", len(comparisons))
    return comparisons
