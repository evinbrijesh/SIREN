"""Multi-basin FNO retrospective validation (ADR-012 §4 load-bearing gate).

Validates the trained FNO-2D hydrodynamic surrogate against documented
arrival times from three Himalayan GLOF/avalanche events:

    1. South Lhonak GLOF (Oct 2023, Teesta) — already validated, 12.3% MAPE
    2. Chamoli rock-ice avalanche (Feb 2021, Rishiganga/Dhauliganga)
    3. Dig Tsho GLOF (Aug 1985, Dudh Koshi/Langmoche Chu)

Gate for load-bearing transition (ADR-012 §4):
    MAPE ≤ 20% on at least 2 of 3 validation events,
    with no single point exceeding 30% error.

Ground truth sources:
    - South Lhonak: documented Teesta valley travel times (3 points)
    - Chamoli: seismic data from Shugar et al. (2021), ICIMOD report
    - Dig Tsho: Vuichard & Zimmermann (1987), Somos-Valenzuela et al. (2015)

Usage:
    python -m siren.ml.eval_multibasin
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from siren.geo.hydro_surrogate import FNO2D, DEFAULT_GRID_SIZE
from siren.ml.train_fno_surrogate import generate_teesta_corridor_dem

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"


# ---------------------------------------------------------------------------
# Event 1: South Lhonak (already validated — reuse parameters)
# ---------------------------------------------------------------------------
SOUTH_LHONAK = {
    "event": "South Lhonak GLOF, Oct 3-4, 2023",
    "corridor": "Teesta River, Sikkim Himalaya",
    "v_breach_mcm": 45.0,
    "q_peak_cms": 20_000,
    "corridor_params": {
        "source_elevation_m": 5200,
        "outlet_elevation_m": 400,
        "distance_km": 85,
        "upper_slope_deg": 8.0,
        "lower_slope_deg": 3.0,
        "grid_size": 64,
        "cell_size_m": 1330,
    },
    "points": [
        {"name": "Chungthang", "distance_km": 35, "t_arrival_min": (65, 75)},
        {"name": "Dikchu", "distance_km": 65, "t_arrival_min": (140, 150)},
        {"name": "Singtam", "distance_km": 85, "t_arrival_min": (170, 190)},
    ],
}


# ---------------------------------------------------------------------------
# Event 2: Chamoli rock-ice avalanche (Feb 7, 2021, Rishiganga/Dhauliganga)
# ---------------------------------------------------------------------------
# Ground truth from Shugar et al. (2021) seismic analysis and ICIMOD report:
#   - Onset: 04:51 UTC (10:21 IST), Feb 7, 2021
#   - 1st debris wave at Tapovan (NTPC) HEP: 05:07 UTC (10:37 IST) = ~16 min
#   - Source: Ronti Peak, ~5,500m asl; Tapovan: ~1,200m asl
#   - Volume: ~27M m³ of rock and ice
#   - Rishiganga HEP (~13 km) hit before Tapovan (~25 km)
#   - Raini village (~15 km) between the two HEPs
# The Rishiganga gorge is extremely steep (upper section ~15°, lower ~5°)
CHAMOLI = {
    "event": "Chamoli rock-ice avalanche, Feb 7, 2021",
    "corridor": "Rishiganga → Dhauliganga, Uttarakhand Himalaya",
    "v_breach_mcm": 27.0,
    "q_peak_cms": 15_000,  # estimated (debris flow, not pure water)
    "corridor_params": {
        "source_elevation_m": 5500,
        "outlet_elevation_m": 1200,
        "distance_km": 30,
        "upper_slope_deg": 15.0,  # very steep upper gorge
        "lower_slope_deg": 5.0,   # wider valley near Tapovan
        "grid_size": 64,
        "cell_size_m": 470,       # 30km / 64 cells
    },
    "points": [
        {"name": "Rishiganga HEP", "distance_km": 13, "t_arrival_min": (7, 10)},
        {"name": "Raini", "distance_km": 15, "t_arrival_min": (9, 12)},
        {"name": "Tapovan HEP", "distance_km": 25, "t_arrival_min": (14, 18)},
    ],
}


# ---------------------------------------------------------------------------
# Event 3: Dig Tsho GLOF (Aug 4, 1985, Dudh Koshi / Langmoche Chu)
# ---------------------------------------------------------------------------
# Ground truth from Vuichard & Zimmermann (1987) and Somos-Valenzuela (2015):
#   - Trigger: Ice avalanche into Dig Tsho (Langmoche) lake, Aug 4, 1985
#   - Release volume: ~5-6M m³ of water
#   - Peak discharge: ~2,350 m³/s at 7 km, ~1,375 m³/s at 27 km
#   - Wave speed: ~4-5 m/s in the upper gorge (Vuichard & Zimmermann 1987)
#   - Destroyed Thami Hydropower station (~10 km downstream)
#   - Dudh Koshi: source ~4,500m, Lukla ~2,800m
DIG_TSHO = {
    "event": "Dig Tsho GLOF, Aug 4, 1985",
    "corridor": "Langmoche Chu → Dudh Koshi, Khumbu Himalaya",
    "v_breach_mcm": 5.5,
    "q_peak_cms": 2_350,
    "corridor_params": {
        "source_elevation_m": 4500,
        "outlet_elevation_m": 2800,
        "distance_km": 30,
        "upper_slope_deg": 10.0,  # steep Langmoche Chu
        "lower_slope_deg": 4.0,   # wider Dudh Koshi valley
        "grid_size": 64,
        "cell_size_m": 470,       # 30km / 64 cells
    },
    "points": [
        {"name": "Thami", "distance_km": 10, "t_arrival_min": (33, 42)},
        {"name": "Ghat", "distance_km": 20, "t_arrival_min": (67, 83)},
        {"name": "Lukla", "distance_km": 30, "t_arrival_min": (100, 125)},
    ],
}


def _run_fno_validation(
    event_config: dict,
    model: FNO2D,
    device: str,
) -> dict:
    """Run FNO validation for a single event.

    Returns dict with points, MAPE, and gate status.
    """
    params = event_config["corridor_params"]
    grid_size = params["grid_size"]

    # Generate corridor DEM
    dem = generate_teesta_corridor_dem(
        grid_size=grid_size,
        source_elev=params["source_elevation_m"],
        outlet_elev=params["outlet_elevation_m"],
        upper_slope=params["upper_slope_deg"],
        lower_slope=params["lower_slope_deg"],
        random_state=42,
    )

    # Normalize DEM
    dem_min, dem_max = float(dem.min()), float(dem.max())
    dem_norm = (dem - dem_min) / (dem_max - dem_min) if dem_max > dem_min else np.zeros_like(dem)

    # Breach volume (log-scale normalization, same as training)
    v_breach = event_config["v_breach_mcm"] * 1e6
    v_norm = float(np.log1p(v_breach) / 20.0)

    # Prepare input
    v_grid = np.full_like(dem_norm, v_norm, dtype=np.float32)
    x = np.stack([dem_norm, v_grid], axis=0)[np.newaxis]  # (1, 2, H, W)
    x_tensor = torch.from_numpy(x).float().to(device)

    # Run inference
    t0 = time.time()
    with torch.no_grad():
        output = model(x_tensor)
    inference_time = time.time() - t0

    h_water = output["h_water"][0, 0].cpu().numpy()

    # Compute T_arrival deterministically from h_water
    source_row, source_col = np.unravel_index(np.argmax(dem), dem.shape)
    rows, cols = np.indices(dem.shape)
    dist_cells = np.sqrt((rows - source_row) ** 2 + (cols - source_col) ** 2)
    cell_size_m = params["cell_size_m"]
    wave_speed = np.sqrt(9.81 * np.maximum(h_water, 0.1))
    wave_speed = np.clip(wave_speed, 7.0, 10.0)
    t_arrival_grid = (dist_cells * cell_size_m / wave_speed / 60.0).astype(np.float32)

    # Map downstream points to grid locations
    total_km = params["distance_km"]
    points = event_config["points"]
    point_coords = {}
    for pt in points:
        frac = pt["distance_km"] / total_km
        row = int(frac * (grid_size - 1))
        col = grid_size // 2
        point_coords[pt["name"]] = (row, col)

    # Extract predicted arrival times
    predicted_arrivals = {}
    for pt in points:
        name = pt["name"]
        if name in point_coords:
            row, col = point_coords[name]
            r0, r1 = max(0, row - 2), min(grid_size, row + 3)
            c0, c1 = max(0, col - 2), min(grid_size, col + 3)
            predicted_arrivals[name] = float(np.mean(t_arrival_grid[r0:r1, c0:c1]))

    # Compute errors
    results = {
        "event": event_config["event"],
        "corridor": event_config["corridor"],
        "breach_params": {
            "v_breach_mcm": event_config["v_breach_mcm"],
            "q_peak_cms": event_config["q_peak_cms"],
        },
        "inference_time_s": round(inference_time, 3),
        "h_water_max_m": round(float(h_water.max()), 2),
        "h_water_mean_m": round(float(h_water.mean()), 4),
        "points": [],
    }

    total_ape = 0.0
    n_points = 0
    max_error = 0.0

    for pt in points:
        name = pt["name"]
        t_min, t_max = pt["t_arrival_min"]
        t_observed = (t_min + t_max) / 2
        t_predicted = predicted_arrivals.get(name, 0.0)

        ape = abs(t_predicted - t_observed) / t_observed * 100
        total_ape += ape
        n_points += 1
        max_error = max(max_error, ape)

        within_range = t_min <= t_predicted <= t_max

        results["points"].append({
            "name": name,
            "distance_km": pt["distance_km"],
            "t_observed_min": t_min,
            "t_observed_max": t_max,
            "t_observed_mid": t_observed,
            "t_predicted": round(t_predicted, 1),
            "abs_error_min": round(abs(t_predicted - t_observed), 1),
            "percentage_error": round(ape, 1),
            "within_documented_range": within_range,
        })

    mape = total_ape / n_points if n_points > 0 else float("inf")
    results["mape"] = round(mape, 1)
    results["max_point_error"] = round(max_error, 1)
    results["gate_passed"] = mape <= 20.0

    return results


def run_multibasin_validation(device: str = "auto") -> dict:
    """Run FNO validation across all three events.

    Returns the combined results with the ADR-012 §4 load-bearing gate check.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt_path = CHECKPOINT_DIR / "fno_hydro_surrogate_v1.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FNO checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FNO2D(
        modes=state["modes"],
        width=state["width"],
        n_points=state["n_points"],
        n_layers=state["n_layers"],
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    logger.info("Loaded FNO checkpoint (val loss: %.4f, epoch %d)",
                state["best_val_loss"], state["best_epoch"])

    events = [SOUTH_LHONAK, CHAMOLI, DIG_TSHO]
    all_results = []

    for event_config in events:
        logger.info("Validating: %s", event_config["event"])
        result = _run_fno_validation(event_config, model, device)
        all_results.append(result)

        print(f"\n  {result['event']}")
        print(f"  {'Point':<18} {'Dist(km)':<10} {'Observed(min)':<16} {'Predicted(min)':<16} {'Error(%)':<10}")
        print(f"  {'-'*18} {'-'*10} {'-'*16} {'-'*16} {'-'*10}")
        for pt in result["points"]:
            obs_str = f"{pt['t_observed_min']}-{pt['t_observed_max']}"
            print(f"  {pt['name']:<18} {pt['distance_km']:<10} "
                  f"{obs_str:<16} {pt['t_predicted']:<16.1f} "
                  f"{pt['percentage_error']:<10.1f}")
        print(f"  MAPE: {result['mape']:.1f}%  —  {'PASS' if result['gate_passed'] else 'FAIL'}")

    # ADR-012 §4 gate: MAPE ≤ 20% on at least 2 of 3, no single point > 30%
    n_passed = sum(1 for r in all_results if r["gate_passed"])
    max_point_error = max(r["max_point_error"] for r in all_results)

    combined = {
        "events": all_results,
        "n_events_passed": n_passed,
        "n_events_total": len(all_results),
        "max_point_error_pct": max_point_error,
        "load_bearing_gate": {
            "requirement": "MAPE ≤ 20% on ≥ 2 of 3 events, no single point > 30%",
            "n_passed": n_passed,
            "max_point_error": max_point_error,
            "passed": n_passed >= 2 and max_point_error <= 30.0,
        },
    }

    print(f"\n{'=' * 70}")
    print("Multi-Basin FNO Validation Summary (ADR-012 §4)")
    print(f"{'=' * 70}")
    for r in all_results:
        status = "PASS" if r["gate_passed"] else "FAIL"
        print(f"  {r['event']:<50} MAPE={r['mape']:.1f}%  {status}")
    print(f"\n  Events passed: {n_passed}/{len(all_results)}")
    print(f"  Max point error: {max_point_error:.1f}%")
    print(f"  Load-bearing gate: {'PASS' if combined['load_bearing_gate']['passed'] else 'FAIL'}")
    print(f"{'=' * 70}")

    # Save results
    results_path = CHECKPOINT_DIR / "multibasin_validation.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info("Results saved: %s", results_path)

    return combined


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-basin FNO validation")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_multibasin_validation(device=args.device)


if __name__ == "__main__":
    main()
