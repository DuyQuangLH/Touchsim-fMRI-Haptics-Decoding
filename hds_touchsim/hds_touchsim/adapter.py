"""Adapter between the HDS and TouchSim (https://github.com/hsaal/touchsim).

The TouchSim hand is ~209 mm long; the real hand is scaled to REAL_HAND_LENGTH_MM. The HDS keeps
its real size, so positions and pin radius are converted to TouchSim mm by dividing by hand_scale.
All functions take the `touchsim` module as `ts`, so this file imports without TouchSim installed.
"""
import numpy as np

from .hds import HDSSpec, bounding_box_px, pin_positions_px

PALM_REGIONS = ["Pw1_p", "Pw2_p", "Pw3_p", "Pw4_p", "Pp1_p", "Pp2_p"]
REAL_HAND_LENGTH_MM = 189.0   # mean hand length, wrist to middle fingertip
CLASS_RATIO = {"SA1": 2, "RA": 4, "PC": 1}   # Saal et al. 2017: SA1 = 2x PC, RA = 2x SA1


def load_touchsim():
    import touchsim as ts
    return ts


def ts_px_per_mm(ts) -> float:
    """TouchSim's own scale (px per TouchSim mm, ~2.18). Never overwrite it."""
    return float(ts.hand_surface.pxl_per_mm)


def ts_hand_length_mm(ts) -> float:
    """Length of the TouchSim hand outline in TouchSim mm (~209)."""
    b = np.vstack(ts.hand_surface.boundary) / ts_px_per_mm(ts)
    return float(np.ptp(b[:, 1]))


def hand_scale(ts) -> float:
    """Uniform factor (<1) shrinking the TouchSim hand to the real length."""
    return REAL_HAND_LENGTH_MM / ts_hand_length_mm(ts)


def px_per_mm(ts) -> float:
    """Pixels per real mm on the (fixed) hand image."""
    return ts_px_per_mm(ts) / hand_scale(ts)


def pins_px(ts, spec: HDSSpec, center_px):
    """(rows, cols, 2) pin centers in pixel space."""
    return pin_positions_px(spec, center_px, px_per_mm(ts))


def pins_hand_mm(ts, spec: HDSSpec, center_px):
    """(rows, cols, 2) pin centers in TouchSim hand space (mm)."""
    p = pins_px(ts, spec, center_px)
    flat = ts.hand_surface.pixel2hand(p.reshape(-1, 2))
    return flat.reshape(p.shape)


def region_mask_fn(ts, tags=PALM_REGIONS, box=(0, 450, 0, 560)):
    """Returns f(x_px, y_px) -> bool: is the pixel inside any of the named hand regions?
    Rasterised once at 1 px so the position search is fast."""
    hs = ts.hand_surface
    x0, x1, y0, y1 = box
    xs, ys = np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts_px = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    _, ids = hs.locate(hs.pixel2hand(pts_px))
    want = [hs.tags.index(t) for t in tags]
    inside = np.isin(ids, want).reshape(gx.shape)

    def f(x, y):
        i, j = int(round(x)) - x0, int(round(y)) - y0
        return 0 <= i < inside.shape[0] and 0 <= j < inside.shape[1] and bool(inside[i, j])
    return f


def build_stimulus(ts, tensor, spec: HDSSpec, center_px, duration_s=None, ramp_s=None, fs=5000):
    duration_s = spec.duration_s if duration_s is None else duration_s
    ramp_s = spec.ramp_s if ramp_s is None else ramp_s
    hand = pins_hand_mm(ts, spec, center_px)
    pin_radius_ts = spec.pin_diameter_mm / 2 / hand_scale(ts)   # real mm -> TouchSim mm
    stim = None
    for r in range(tensor.shape[0]):
        for c in range(tensor.shape[1]):
            if not tensor[r, c]:
                continue
            s = ts.stim_ramp(amp=spec.depth_mm, len=duration_s, loc=hand[r, c],
                             ramp_len=ramp_s, fs=fs, pin_radius=pin_radius_ts)
            if stim is None:
                stim = s
            else:
                stim += s
    if stim is None:
        raise ValueError("Tensor has no active pins.")
    return stim


def balance_classes(pop, ratio=CLASS_RATIO, seed=0):
    """Randomly subsample afferents so the class counts follow `ratio` (as many as possible)."""
    cls = np.array(pop.affclass)
    unit = min((cls == k).sum() / w for k, w in ratio.items())
    rng = np.random.default_rng(seed)
    sel = np.zeros(len(cls), bool)
    for k, w in ratio.items():
        sel[rng.choice(np.where(cls == k)[0], int(round(unit * w)), replace=False)] = True
    return pop[sel]


def make_population(ts, spec: HDSSpec, center_px, margin_mm=20.0, density_multiplier=1.0,
                    class_ratio=CLASS_RATIO, seed=0):
    """Afferents (SA1, RA, PC) under / near the HDS box (+ margin), class-balanced.
    Build once per HDS position and reuse for every digit so responses are comparable."""
    pop = ts.affpop_hand(density_multiplier=density_multiplier)
    ppm = px_per_mm(ts)
    loc_px = ts.hand_surface.hand2pixel(pop.location)
    x0, y0, x1, y1 = bounding_box_px(spec, center_px, ppm)
    m = margin_mm * ppm
    keep = ((loc_px[:, 0] >= x0 - m) & (loc_px[:, 0] <= x1 + m) &
            (loc_px[:, 1] >= y0 - m) & (loc_px[:, 1] <= y1 + m))
    if not keep.any():
        raise ValueError("No afferents near the HDS - is the center on the hand?")
    pop = pop[keep]
    return balance_classes(pop, class_ratio, seed) if class_ratio else pop


def respond(ts, stim, pop):
    return pop.response(stim)
