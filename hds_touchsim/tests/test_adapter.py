import numpy as np
import pytest

from fake_touchsim import HAND_LEN_MM, PPM, make_fake_touchsim
from hds_touchsim import (HDSSpec, balance_classes, build_stimulus, digit_tensor, fit_to_region,
                          hand_scale, make_population, px_per_mm, region_mask_fn,
                          similarity_matrix, spike_counts)
from hds_touchsim.adapter import REAL_HAND_LENGTH_MM, pins_hand_mm, ts_hand_length_mm

ts = make_fake_touchsim()
spec = HDSSpec()
CENTER = (230.0, 200.0)


def test_hand_scale_and_ppm():
    assert np.isclose(ts_hand_length_mm(ts), HAND_LEN_MM)
    assert np.isclose(hand_scale(ts), REAL_HAND_LENGTH_MM / HAND_LEN_MM)
    assert np.isclose(px_per_mm(ts), PPM / hand_scale(ts))


def test_hds_is_not_scaled_with_hand_in_pixels():
    p = pins_hand_mm(ts, spec, CENTER)
    pitch_ts = np.linalg.norm(p[0, 1] - p[0, 0])
    assert np.isclose(pitch_ts, spec.pitch_mm / hand_scale(ts))
    # real size: pixel pitch / px-per-real-mm == 15 mm
    assert np.isclose(pitch_ts * PPM / px_per_mm(ts), spec.pitch_mm)


def test_pin_radius_follows_position_scaling():
    build_stimulus(ts, digit_tensor(8), spec, CENTER)
    assert np.isclose(ts.last_pin_radius, 5.0 / hand_scale(ts))


def test_stimulus_has_one_pin_per_active_cell():
    for d in range(10):
        assert len(build_stimulus(ts, digit_tensor(d), spec, CENTER).pins) == digit_tensor(d).sum()
    with pytest.raises(ValueError):
        build_stimulus(ts, np.zeros((5, 3), np.uint8), spec, CENTER)


def test_balance_classes_ratio():
    pop = ts.affpop_hand()
    out = balance_classes(pop)
    cls = np.array(out.affclass)
    n = {k: int((cls == k).sum()) for k in ("SA1", "RA", "PC")}
    assert abs(n["SA1"] - 2 * n["PC"]) <= 1 and abs(n["RA"] - 4 * n["PC"]) <= 2
    assert (np.array(balance_classes(pop, seed=0).location) == np.array(out.location)).all()


def test_population_near_hds_only():
    pop = make_population(ts, spec, CENTER, class_ratio=None)
    px = ts.hand_surface.hand2pixel(pop.location)
    ppm = px_per_mm(ts)
    half_w, half_h = (25 + 20) * ppm, (40 + 20) * ppm
    assert px[:, 0].min() >= CENTER[0] - half_w - 1e-6 and px[:, 0].max() <= CENTER[0] + half_w + 1e-6
    assert px[:, 1].min() >= CENTER[1] - half_h - 1e-6 and px[:, 1].max() <= CENTER[1] + half_h + 1e-6


def test_fit_inside_palm_and_pipeline_smoke():
    ppm = px_per_mm(ts)
    best, n, _ = fit_to_region(spec, region_mask_fn(ts), (40, 400, 100, 400), pxl_per_mm=ppm, step_px=10)
    assert n >= 1
    pop = make_population(ts, spec, best)
    counts = {d: spike_counts(ts_resp) for d, ts_resp in
              ((d, pop.response(build_stimulus(ts, digit_tensor(d), spec, best))) for d in range(10))}
    S = similarity_matrix(counts)
    assert S.shape == (10, 10) and np.allclose(np.diag(S), 1.0) and np.allclose(S, S.T)
