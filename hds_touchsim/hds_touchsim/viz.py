"""Minimal plotting: hand outline with the HDS on top (placement check)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from .adapter import PALM_REGIONS, px_per_mm
from .hds import HDSSpec, bounding_box_px, digit_tensor, pin_positions_px


def plot_layout(ts, center_px, digit=3, spec=None, highlight=PALM_REGIONS, ax=None):
    """Hand (palm regions shaded) and the HDS pattern of `digit`, axes in real mm."""
    spec = spec or HDSSpec()
    hs, ppm = ts.hand_surface, px_per_mm(ts)
    ax = ax or plt.subplots(figsize=(5, 6))[1]
    for b, poly in enumerate(hs.boundary):
        ax.add_patch(Polygon(poly / ppm, closed=True, ec="k", lw=1,
                             fc="#fde9c9" if hs.tags[b] in highlight else "none"))
    x0, y0, x1, y1 = np.array(bounding_box_px(spec, center_px, ppm)) / ppm
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="tab:blue", lw=1.5))
    t, pos = digit_tensor(digit), pin_positions_px(spec, center_px, ppm) / ppm
    for r in range(spec.rows):
        for c in range(spec.cols):
            ax.add_patch(Circle(pos[r, c], spec.pin_diameter_mm / 2,
                                fc="tab:red" if t[r, c] else "none", ec="k", alpha=.8, lw=.8))
    allb = np.vstack(hs.boundary) / ppm
    ax.set_xlim(allb[:, 0].min() - 5, allb[:, 0].max() + 5)
    ax.set_ylim(allb[:, 1].min() - 5, allb[:, 1].max() + 5)
    ax.set_aspect("equal"); ax.set_xlabel("mm"); ax.set_title(f"HDS digit {digit}")
    return ax
