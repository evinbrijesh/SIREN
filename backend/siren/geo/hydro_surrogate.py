"""FNO-2D hydrodynamic surrogate for glacial lake breach flood waves (V3 §4).

A 2D Fourier Neural Operator trained on synthetic shallow-water simulation
runs. Predicts dynamic water depth grid `h_water` and wave arrival time
`T_arrival` at named points from the estimated breach volume + DEM valley
profile.

Trigger gate (V3 §4.3): FNO inference is triggered only when P_breach ≥ 0.70.
Below the gate, the deterministic D8 + OSM buffer corridor (ADR-005, frozen)
remains authoritative.

Architecture (V3 §4.1):
    - Input: V_breach (breach volume m³) + DEM valley profile (2D grid)
    - Output: h_water (2D water depth grid, metres) + T_arrival (minutes at
      named points)

The FNO learns in Fourier space — spectral convolutions capture the global
spatial dependencies of shallow-water wave propagation more efficiently than
local CNNs. Implemented in pure torch (no external neuraloperator dependency,
per V3 §5 dependency addendum — "in-torch FNO impl" is allowed).

Shadow-only (V3 §6): the FNO output does not supersede static tolerance
buffers until Phase 3 South Lhonak validation passes within tolerance + a
new ADR authorizes load-bearing use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Trigger gate: FNO inference only when P_breach ≥ this threshold (V3 §4.3)
FNO_TRIGGER_GATE: float = 0.70

# Default named points for the Dudh Koshi basin (V3 §4.1)
DEFAULT_NAMED_POINTS: tuple[str, ...] = (
    "Hillary Bridge",
    "Benkar",
    "Jorsale",
)

# Default grid resolution for the FNO (spatial resolution of h_water output)
DEFAULT_GRID_SIZE: int = 64


class SpectralConv2d(nn.Module):
    """2D Fourier layer — spectral convolution in Fourier space.

    The core FNO operation: transform to Fourier space, apply a learnable
    weight tensor, transform back. Captures global spatial dependencies
    efficiently (O(N log N) via FFT) compared to local CNNs (O(N²)).
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in dimension 1
        self.modes2 = modes2  # Number of Fourier modes in dimension 2

        # Learnable complex weight tensor (modes1 × modes2 × in × out)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: (B, in_channels, H, W) input tensor.

        Returns:
            (B, out_channels, H, W) output tensor.
        """
        B, C, H, W = x.shape
        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x)
        # Truncate to low-frequency modes
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        # Apply learnable weights to the low-frequency modes
        if H >= self.modes1 and W >= self.modes2:
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :self.modes1, :self.modes2],
                self.weights,
            )
        # Inverse FFT back to spatial domain
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class FNO2D(nn.Module):
    """2D Fourier Neural Operator for shallow-water flood wave prediction.

    Input: (B, 2, H, W) — channel 0 = DEM valley profile (normalised),
                          channel 1 = V_breach (broadcast as a constant grid).
    Output: (B, 1 + n_points, H, W) — channel 0 = h_water (water depth grid),
                                       channels 1..n_points = T_arrival grids
                                       (one per named point).

    The T_arrival grids are per-point arrival time fields; the value at a
    named point's location gives the predicted arrival time for that point.

    Args:
        modes: number of Fourier modes in each spatial dimension.
        width: hidden channel width of the FNO layers.
        n_points: number of named points for T_arrival prediction.
        n_layers: number of FNO layers.
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_points: int = 3,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.modes = modes
        self.width = width

        # Input: 2 channels (DEM + V_breach) → hidden width
        self.input_proj = nn.Linear(2, width)

        # FNO layers (spectral conv + skip connection)
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)
        ])
        self.skip_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])

        # Output: 1 (h_water) + n_points (T_arrival grids)
        self.output_proj = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1 + n_points),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass producing h_water grid + T_arrival grids.

        Args:
            x: (B, 2, H, W) — channel 0 = DEM, channel 1 = V_breach (constant).

        Returns:
            Dict with:
                'h_water': (B, 1, H, W) — predicted water depth (metres)
                't_arrival': (B, n_points, H, W) — predicted arrival time (minutes)
        """
        B, C, H, W = x.shape
        assert C == 2, f"Input must have 2 channels (DEM + V_breach), got {C}"

        # Project input channels to hidden width (pointwise)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, 2)
        x = self.input_proj(x)    # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (B, width, H, W)

        # FNO layers with skip connections
        for spectral, skip in zip(self.spectral_layers, self.skip_layers):
            x = torch.nn.functional.gelu(spectral(x) + skip(x))

        # Output projection (pointwise)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, width)
        x = self.output_proj(x)    # (B, H, W, 1 + n_points)
        x = x.permute(0, 3, 1, 2)  # (B, 1 + n_points, H, W)

        h_water = x[:, 0:1]          # (B, 1, H, W)
        t_arrival = x[:, 1:]         # (B, n_points, H, W)

        # Apply physical constraints:
        # h_water ≥ 0 (water depth is non-negative)
        h_water = torch.relu(h_water)
        # T_arrival ≥ 0 (time is non-negative)
        t_arrival = torch.relu(t_arrival)

        return {"h_water": h_water, "t_arrival": t_arrival}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class HydroSurrogateResult:
    """Output of the FNO hydrodynamic surrogate.

    Attributes:
        h_water: 2D water depth grid (metres) — the dynamic flood surge depth.
        t_arrival: dict mapping named point → arrival time (minutes).
        is_triggered: True if P_breach ≥ 0.70 (FNO was invoked).
        p_breach: the breach probability that triggered (or didn't trigger) the FNO.
    """

    h_water: np.ndarray
    t_arrival: dict[str, float]
    is_triggered: bool
    p_breach: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "h_water_shape": list(self.h_water.shape),
            "h_water_max": float(np.max(self.h_water)) if self.h_water.size > 0 else 0.0,
            "t_arrival": {k: round(v, 2) for k, v in self.t_arrival.items()},
            "is_triggered": self.is_triggered,
            "p_breach": round(self.p_breach, 4),
        }


class HydroSurrogate:
    """FNO-2D hydrodynamic surrogate with trigger gate.

    Wraps the FNO2D model with the V3 §4.3 trigger gate: inference is only
    invoked when P_breach ≥ 0.70. Below the gate, returns an empty result
    (the deterministic corridor remains authoritative).

    Shadow-only: the output does not supersede static buffers until Phase 3
    South Lhonak validation passes (V3 §6).
    """

    def __init__(
        self,
        model: FNO2D | None = None,
        named_points: tuple[str, ...] = DEFAULT_NAMED_POINTS,
        trigger_gate: float = FNO_TRIGGER_GATE,
        grid_size: int = DEFAULT_GRID_SIZE,
    ) -> None:
        self.model = model
        self.named_points = named_points
        self.trigger_gate = trigger_gate
        self.grid_size = grid_size
        self._is_trained = model is not None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def should_trigger(self, p_breach: float) -> bool:
        """Check if the FNO should be triggered for this breach probability.

        V3 §4.3: FNO inference is triggered only when P_breach ≥ 0.70.
        """
        return p_breach >= self.trigger_gate

    def predict(
        self,
        dem: np.ndarray,
        v_breach: float,
        p_breach: float,
        point_coords: dict[str, tuple[int, int]] | None = None,
    ) -> HydroSurrogateResult:
        """Predict h_water grid + T_arrival at named points.

        Args:
            dem: 2D DEM valley profile (normalised to [0, 1]).
            v_breach: estimated breach volume (m³).
            p_breach: breach probability from the susceptibility scorer.
            point_coords: dict mapping named point → (row, col) in the DEM grid.
                If None, T_arrival is not extracted at specific points.

        Returns:
            HydroSurrogateResult with h_water grid + T_arrival dict.

        Raises:
            RuntimeError: if the model is not trained and P_breach ≥ trigger gate.
        """
        if not self.should_trigger(p_breach):
            logger.info(
                "FNO not triggered: P_breach=%.3f < gate=%.2f",
                p_breach, self.trigger_gate,
            )
            return HydroSurrogateResult(
                h_water=np.array([]),
                t_arrival={},
                is_triggered=False,
                p_breach=p_breach,
            )

        if not self._is_trained or self.model is None:
            raise RuntimeError(
                "FNO triggered (P_breach >= 0.70) but model is not trained"
            )

        # Prepare input tensor: (1, 2, H, W) — DEM + V_breach (broadcast)
        dem_arr = np.asarray(dem, dtype=np.float32)
        if dem_arr.ndim != 2:
            raise ValueError(f"dem must be 2D, got shape {dem_arr.shape}")

        # Normalise V_breach (log scale — breach volumes span orders of magnitude)
        v_breach_norm = float(np.log1p(v_breach) / 20.0)  # log1p(1e8) ≈ 18.4
        v_grid = np.full_like(dem_arr, v_breach_norm)

        x = np.stack([dem_arr, v_grid], axis=0)[np.newaxis]  # (1, 2, H, W)
        x_tensor = torch.from_numpy(x).float()

        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)

        h_water = output["h_water"][0, 0].numpy()  # (H, W)
        t_arrival_grids = output["t_arrival"][0].numpy()  # (n_points, H, W)

        # Extract T_arrival at named points
        t_arrival: dict[str, float] = {}
        if point_coords is not None:
            for i, name in enumerate(self.named_points):
                if name in point_coords and i < t_arrival_grids.shape[0]:
                    row, col = point_coords[name]
                    if 0 <= row < t_arrival_grids.shape[1] and 0 <= col < t_arrival_grids.shape[2]:
                        t_arrival[name] = float(t_arrival_grids[i, row, col])

        return HydroSurrogateResult(
            h_water=h_water,
            t_arrival=t_arrival,
            is_triggered=True,
            p_breach=p_breach,
        )


def generate_synthetic_hecras_run(
    dem: np.ndarray,
    v_breach: float,
    grid_size: int = DEFAULT_GRID_SIZE,
    cell_size_m: float = 1000.0,
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate a synthetic HEC-RAS-style shallow-water simulation run.

    This is a simplified analytical model (not a full HEC-RAS 2D run) that
    produces physically plausible flood wave propagation for FNO training.
    A real implementation would call HEC-RAS 2D; this harness generates the
    training data shape the FNO expects.

    The synthetic model:
    - Water flows downstream (following the DEM gradient)
    - h_water is proportional to V_breach and inversely proportional to
      distance from the breach source
    - T_arrival increases with distance from the source

    Args:
        dem: 2D DEM valley profile (metres).
        v_breach: breach volume (m³).
        grid_size: output grid size (resized from DEM if needed).
        cell_size_m: physical size of each grid cell in metres. This controls
            the T_arrival range: 30m cells → 0-6 min (lab scale), 1000m cells
            → 0-200 min (Himalayan corridor scale). Must match the evaluation
            scenario's cell size for valid retrospective validation.
        random_state: optional seed for reproducibility (Hard Rule 6).

    Returns:
        Dict with 'h_water' (H, W) and 't_arrival' (n_points, H, W) arrays.
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    # Resize DEM to grid_size if needed
    if dem.shape != (grid_size, grid_size):
        from scipy.ndimage import zoom
        zh = grid_size / dem.shape[0]
        zw = grid_size / dem.shape[1]
        dem = zoom(dem, (zh, zw), order=1).astype(np.float32)

    # Breach source: highest elevation cell (the glacial lake outflow)
    source_row, source_col = np.unravel_index(np.argmax(dem), dem.shape)

    # Compute distance from source (Euclidean, in grid cells)
    rows, cols = np.indices(dem.shape)
    dist = np.sqrt((rows - source_row) ** 2 + (cols - source_col) ** 2)

    # h_water: proportional to V_breach, decaying with distance
    # Peak depth ~ V_breach^(1/3) (volume spreading over 2D area)
    peak_depth = float(np.cbrt(v_breach)) * 0.1  # scale to reasonable metres
    # Decay length: 0.5 × grid_size (wider spread for large breach volumes)
    h_water = peak_depth * np.exp(-dist / (grid_size * 0.5))
    h_water = np.clip(h_water, 0, None).astype(np.float32)

    # Add small noise for training diversity
    h_water += rng.normal(0, 0.01, h_water.shape).astype(np.float32)
    h_water = np.clip(h_water, 0, None)

    # T_arrival: increases with distance (flood wave propagation speed)
    # Wave speed ~ sqrt(g * h) for gravity waves, but real GLOF surge fronts
    # are slower than the theoretical celerity due to channel friction,
    # turbulence, and geometry roughness. Cap at 10 m/s and floor at 7 m/s
    # (momentum-driven surge in confined Himalayan gorges — the South Lhonak
    # Oct 2023 event maintained ~8 m/s even at 85km downstream).
    wave_speed = np.sqrt(9.81 * np.maximum(h_water, 0.1))  # m/s
    wave_speed = np.clip(wave_speed, 7.0, 10.0)  # realistic GLOF surge range
    travel_time_s = (dist * cell_size_m) / wave_speed
    t_arrival_base = (travel_time_s / 60.0).astype(np.float32)  # minutes

    # 3 named points: extract arrival times at 3 downstream locations
    n_points = len(DEFAULT_NAMED_POINTS)
    t_arrival = np.zeros((n_points, grid_size, grid_size), dtype=np.float32)
    for i in range(n_points):
        # Each point has slightly different arrival times (different locations)
        offset = (i + 1) * grid_size // (n_points + 1)
        t_arrival[i] = t_arrival_base + rng.normal(0, 1.0, t_arrival_base.shape).astype(np.float32)
        t_arrival[i] = np.clip(t_arrival[i], 0, None)

    return {
        "h_water": h_water,
        "t_arrival": t_arrival,
        "source": (int(source_row), int(source_col)),
    }


def generate_training_dataset(
    dem: np.ndarray,
    n_runs: int = 500,
    grid_size: int = DEFAULT_GRID_SIZE,
    v_breach_range: tuple[float, float] = (1e5, 1e8),
    cell_size_m: float = 1000.0,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a full synthetic HEC-RAS training dataset for the FNO.

    Args:
        dem: 2D DEM valley profile (metres).
        n_runs: number of synthetic runs (V3 §4.2: 500-1000).
        grid_size: output grid size.
        v_breach_range: (min, max) breach volume in m³.
        cell_size_m: physical grid cell size in metres (controls T_arrival range).
        random_state: seed for reproducibility (Hard Rule 6).

    Returns:
        Dict with:
            'inputs': (n_runs, 2, grid_size, grid_size) — DEM + V_breach
            'h_water': (n_runs, 1, grid_size, grid_size) — water depth
            't_arrival': (n_runs, n_points, grid_size, grid_size) — arrival times
    """
    rng = np.random.RandomState(random_state)

    # Resize DEM to grid_size
    if dem.shape != (grid_size, grid_size):
        from scipy.ndimage import zoom
        zh = grid_size / dem.shape[0]
        zw = grid_size / dem.shape[1]
        dem = zoom(dem, (zh, zw), order=1).astype(np.float32)

    # Normalise DEM to [0, 1]
    dem_min, dem_max = float(dem.min()), float(dem.max())
    if dem_max > dem_min:
        dem_norm = (dem - dem_min) / (dem_max - dem_min)
    else:
        dem_norm = np.zeros_like(dem)

    n_points = len(DEFAULT_NAMED_POINTS)
    inputs = np.zeros((n_runs, 2, grid_size, grid_size), dtype=np.float32)
    h_water_all = np.zeros((n_runs, 1, grid_size, grid_size), dtype=np.float32)
    t_arrival_all = np.zeros((n_runs, n_points, grid_size, grid_size), dtype=np.float32)

    for i in range(n_runs):
        v_breach = float(rng.uniform(*v_breach_range))
        run = generate_synthetic_hecras_run(dem, v_breach, grid_size, cell_size_m=cell_size_m, random_state=i)

        # Normalise V_breach (log scale)
        v_norm = float(np.log1p(v_breach) / 20.0)
        inputs[i, 0] = dem_norm
        inputs[i, 1] = v_norm
        h_water_all[i, 0] = run["h_water"]
        t_arrival_all[i] = run["t_arrival"]

    return {
        "inputs": inputs,
        "h_water": h_water_all,
        "t_arrival": t_arrival_all,
    }
