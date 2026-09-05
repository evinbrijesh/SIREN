"""Generate visual heatmap PNGs from change masks and confidence arrays.

Produces colorized PNGs for the UI:
  - Change heatmap: red gradient over changed pixels, dark background elsewhere
  - Confidence heatmap: green→yellow→red gradient over confidence values
  - Before/after comparison: side-by-side or overlay with change highlighted

These PNGs are served by the existing /data/processed/ static mount and
displayed in the ReviewView and MapView UI components.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_change_heatmap_png(
    mask: np.ndarray,
    output_path: Path | str,
    confidence: np.ndarray | None = None,
) -> Path:
    """Generate a colorized change heatmap PNG from a binary mask.

    The heatmap uses a red gradient for changed pixels, with intensity
    modulated by the confidence array if provided. Unchanged pixels are
    dark (near-black with a subtle blue tint for the dark UI theme).

    Args:
        mask: Binary change mask (H, W) — 1 = changed
        output_path: Where to save the PNG
        confidence: Optional confidence map (H, W) in [0, 1] to modulate intensity

    Returns:
        Path to the saved PNG
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Background: dark slate (#0F172A) matching the UI canvas
    rgb[:, :, 0] = 15   # R
    rgb[:, :, 1] = 23   # G
    rgb[:, :, 2] = 42   # B

    changed = mask > 0
    if confidence is not None:
        conf = np.clip(confidence, 0, 1)
    else:
        conf = np.ones_like(mask, dtype=np.float32)

    # Changed pixels: red gradient based on confidence
    # High confidence → bright red (#EF4444), low → orange (#F59E0B)
    rgb[changed, 0] = (239 + (1 - conf[changed]) * 0).astype(np.uint8)   # R: 239
    rgb[changed, 1] = (68 + (1 - conf[changed]) * 20).astype(np.uint8)   # G: 68-88
    rgb[changed, 2] = (68 + (1 - conf[changed]) * 0).astype(np.uint8)    # B: 68

    # Add a subtle glow border around changed regions (dilation effect)
    glow = _dilate_mask(changed.astype(np.uint8), iterations=2)
    glow_only = (glow > 0) & ~changed
    rgb[glow_only, 0] = np.minimum(rgb[glow_only, 0].astype(int) + 30, 255).astype(np.uint8)
    rgb[glow_only, 1] = np.minimum(rgb[glow_only, 1].astype(int) + 10, 255).astype(np.uint8)
    rgb[glow_only, 2] = np.minimum(rgb[glow_only, 2].astype(int) + 10, 255).astype(np.uint8)

    _save_png(rgb, output_path)
    logger.info(f"Generated change heatmap: {output_path}")
    return output_path


def generate_confidence_heatmap_png(
    confidence: np.ndarray,
    output_path: Path | str,
) -> Path:
    """Generate a confidence heatmap PNG with a green→yellow→red gradient.

    Args:
        confidence: Confidence map (H, W) in [0, 1]
        output_path: Where to save the PNG

    Returns:
        Path to the saved PNG
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = confidence.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Background: dark
    rgb[:, :] = [15, 23, 42]

    conf = np.clip(confidence, 0, 1)

    # Green (safe) → Yellow (watch) → Orange (elevated) → Red (critical)
    # 0.0 = #22C55E, 0.3 = #F59E0B, 0.6 = #F97316, 1.0 = #EF4444
    for y in range(h):
        for x in range(w):
            c = conf[y, x]
            if c < 0.001:
                continue
            rgb[y, x] = _confidence_color(c)

    _save_png(rgb, output_path)
    logger.info(f"Generated confidence heatmap: {output_path}")
    return output_path


def generate_before_after_png(
    baseline_mask: np.ndarray,
    current_mask: np.ndarray,
    output_path: Path | str,
) -> Path:
    """Generate a before/after comparison PNG.

    Shows baseline water extent in cyan and current expansion in red,
    overlaid on a dark background. This gives a clear visual of what
    changed between the two observations.

    Args:
        baseline_mask: Binary water mask at T0 (H, W)
        current_mask: Binary water mask at T1 (H, W)
        output_path: Where to save the PNG

    Returns:
        Path to the saved PNG
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = baseline_mask.shape[:2]
    current_resized = _resize_mask(current_mask, (h, w))

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = [15, 23, 42]  # dark background

    # Baseline water: cyan (#06B6D4)
    baseline = baseline_mask > 0
    rgb[baseline, 0] = 6
    rgb[baseline, 1] = 182
    rgb[baseline, 2] = 212

    # Expansion (in current but not baseline): red (#EF4444)
    expansion = (current_resized > 0) & ~baseline
    rgb[expansion, 0] = 239
    rgb[expansion, 1] = 68
    rgb[expansion, 2] = 68

    # Recession (in baseline but not current): dim orange
    recession = baseline & (current_resized == 0)
    rgb[recession, 0] = 100
    rgb[recession, 1] = 50
    rgb[recession, 2] = 20

    _save_png(rgb, output_path)
    logger.info(f"Generated before/after comparison: {output_path}")
    return output_path


def _confidence_color(c: float) -> tuple[int, int, int]:
    """Map a confidence value [0, 1] to an RGB color."""
    if c < 0.3:
        # Green to yellow
        t = c / 0.3
        return (int(34 + (245 - 34) * t), int(197 + (158 - 197) * t), int(94 + (11 - 94) * t))
    elif c < 0.6:
        # Yellow to orange
        t = (c - 0.3) / 0.3
        return (int(245 + (249 - 245) * t), int(158 + (115 - 158) * t), int(11 + (22 - 11) * t))
    else:
        # Orange to red
        t = (c - 0.6) / 0.4
        return (int(249 + (239 - 249) * t), int(115 + (68 - 115) * t), int(22 + (68 - 22) * t))


def _dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Simple binary dilation without scipy/cv2 dependency."""
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        dilated = np.zeros_like(result)
        dilated = (
            (padded[1:-1, 1:-1] > 0) |
            (padded[:-2, 1:-1] > 0) |
            (padded[2:, 1:-1] > 0) |
            (padded[1:-1, :-2] > 0) |
            (padded[1:-1, 2:] > 0)
        ).astype(np.uint8)
        result = dilated
    return result


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask using nearest-neighbor."""
    h, w = target_shape
    if mask.shape[:2] == (h, w):
        return mask
    src_h, src_w = mask.shape[:2]
    row_idx = (np.arange(h) * src_h / h).astype(int)
    col_idx = (np.arange(w) * src_w / w).astype(int)
    return mask[np.ix_(row_idx, col_idx)]


def _save_png(rgb: np.ndarray, path: Path | str) -> None:
    """Save an RGB array as a PNG using rasterio (already in the dependency whitelist)."""
    import rasterio
    from rasterio.transform import from_bounds

    h, w = rgb.shape[:2]
    # Use a simple identity transform; the PNG is for display only
    transform = from_bounds(0, 0, w, h, w, h)

    with rasterio.open(
        str(path), "w", driver="PNG",
        height=h, width=w,
        count=3, dtype="uint8",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)
