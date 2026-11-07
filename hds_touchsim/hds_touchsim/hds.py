"""HDS (5x3 pin display): specification, digit patterns and geometry. No TouchSim dependency."""
from dataclasses import dataclass

import numpy as np

ROWS, COLS = 5, 3


@dataclass(frozen=True)
class HDSSpec:
    pin_diameter_mm: float = 10.0
    pitch_mm: float = 15.0
    depth_mm: float = 1.0
    pressure_mpa: float = 0.3   # not used by the simulation
    ramp_s: float = 0.2
    hold_s: float = 2.6
    width_mm: float = 50.0
    height_mm: float = 80.0
    rows: int = ROWS
    cols: int = COLS

    @property
    def duration_s(self) -> float:
        return 2 * self.ramp_s + self.hold_s

    def pin_offsets_mm(self) -> np.ndarray:
        """(rows, cols, 2) pin centers in mm relative to the display center (x right, y up)."""
        xs = (np.arange(self.cols) - (self.cols - 1) / 2) * self.pitch_mm
        ys = ((self.rows - 1) / 2 - np.arange(self.rows)) * self.pitch_mm  # row 0 on top
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx, gy], axis=-1)


_GLYPHS = {
    0: ["111", "101", "101", "101", "111"],
    1: ["010", "010", "010", "010", "010"],
    2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "111", "001", "111"],
    4: ["101", "101", "111", "001", "001"],
    5: ["111", "100", "111", "001", "111"],
    6: ["100", "100", "111", "101", "111"],
    7: ["111", "001", "001", "001", "001"],
    8: ["111", "101", "111", "101", "111"],
    9: ["111", "101", "111", "001", "001"],
}


def digit_tensor(d: int) -> np.ndarray:
    """Binary tensor (5, 3) of dtype uint8 for digit d."""
    return np.array([[int(c) for c in row] for row in _GLYPHS[d]], dtype=np.uint8)


def hamming_matrix() -> np.ndarray:
    """(10, 10) number of differing pins between every pair of digits."""
    return np.array([[int((digit_tensor(a) != digit_tensor(b)).sum()) for b in range(10)]
                     for a in range(10)])


def pin_positions_px(spec: HDSSpec, center_px, pxl_per_mm: float = 1.0) -> np.ndarray:
    """(rows, cols, 2) pin centers in hand-surface pixels."""
    return np.asarray(center_px, float) + spec.pin_offsets_mm() * pxl_per_mm


def bounding_box_px(spec: HDSSpec, center_px, pxl_per_mm: float = 1.0):
    """Outer box of the display: (xmin, ymin, xmax, ymax) in px."""
    cx, cy = center_px
    return (cx - spec.width_mm / 2 * pxl_per_mm, cy - spec.height_mm / 2 * pxl_per_mm,
            cx + spec.width_mm / 2 * pxl_per_mm, cy + spec.height_mm / 2 * pxl_per_mm)


def fit_to_region(spec: HDSSpec, region_mask_fn, search_box, pxl_per_mm=1.0, step_px=5):
    """Search for the display center that keeps the most pins inside a region.

    region_mask_fn(x, y) -> bool tells whether a hand-surface point (px) lies in the region.
    search_box = (xmin, xmax, ymin, ymax) in px. Returns (best_center, n_inside, grid), where
    grid maps every tried center to its pin count.
    """
    xmin, xmax, ymin, ymax = search_box
    best, best_n, grid = None, -1, {}
    for cx in np.arange(xmin, xmax + 1, step_px):
        for cy in np.arange(ymin, ymax + 1, step_px):
            pts = pin_positions_px(spec, (cx, cy), pxl_per_mm).reshape(-1, 2)
            n = sum(bool(region_mask_fn(x, y)) for x, y in pts)
            grid[(cx, cy)] = n
            if n > best_n:
                best, best_n = (cx, cy), n
    # several centers usually tie: take the one closest to the middle of the tied set,
    # which keeps the display away from the region edges
    tied = np.array([k for k, v in grid.items() if v == best_n], float)
    best = tuple(tied[np.argmin(((tied - tied.mean(axis=0)) ** 2).sum(axis=1))])
    return best, best_n, grid
