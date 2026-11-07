import numpy as np

from hds_touchsim import HDSSpec, digit_tensor, fit_to_region, hamming_matrix
from hds_touchsim.hds import bounding_box_px, pin_positions_px


def test_digits_shape_and_binary():
    for d in range(10):
        t = digit_tensor(d)
        assert t.shape == (5, 3) and set(np.unique(t)) <= {0, 1}
    assert digit_tensor(8).sum() == 13 and digit_tensor(0).sum() == 12


def test_geometry_matches_spec():
    s = HDSSpec()
    p = pin_positions_px(s, (0, 0), 1.0)
    assert np.isclose(p[0, 1, 0] - p[0, 0, 0], 15.0)
    assert np.isclose(p[0, 0, 1] - p[1, 0, 1], 15.0)
    x0, y0, x1, y1 = bounding_box_px(s, (0, 0))
    assert (x1 - x0, y1 - y0) == (50.0, 80.0)
    assert p[..., 0].max() + 5 <= x1 and p[..., 1].max() + 5 <= y1


def test_scale_applies():
    p2 = pin_positions_px(HDSSpec(), (100, 100), 2.0)
    assert np.isclose(p2[0, 1, 0] - p2[0, 0, 0], 30.0)


def test_fit_finds_inside_position():
    inside = lambda x, y: 100 <= x <= 300 and 200 <= y <= 500
    c, n, _ = fit_to_region(HDSSpec(), inside, (80, 320, 180, 520), step_px=10)
    assert n == 15 and 100 <= c[0] <= 300


def test_timing():
    s = HDSSpec()
    assert s.ramp_s == 0.2 and np.isclose(s.duration_s, 3.0) and s.duration_s >= 2 * s.ramp_s


def test_hamming_symmetric_zero_diagonal():
    H = hamming_matrix()
    assert (H == H.T).all() and (np.diag(H) == 0).all()
