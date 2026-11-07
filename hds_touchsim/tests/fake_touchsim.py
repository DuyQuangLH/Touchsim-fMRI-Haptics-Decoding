"""Tiny stand-in for the `touchsim` module, only for testing the adapter without TouchSim."""
import numpy as np

PPM = 2.18
HAND_LEN_MM = 209.0
_TOP = 70 + HAND_LEN_MM * PPM
_ORIGIN = np.array([225.0, 300.0])
_PALM = ["Pw1_p", "Pw2_p", "Pw3_p", "Pw4_p", "Pp1_p", "Pp2_p"]


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


class _HandSurface:
    pxl_per_mm = PPM

    def __init__(self):
        xs = [80, 230, 380]
        self.tags = list(_PALM) + ["D3_d"]
        self._rects = [(xs[i % 2], 70 + 130 * (i // 2), xs[i % 2 + 1], 200 + 130 * (i // 2))
                       for i in range(6)] + [(200, 460, 260, _TOP)]
        self.boundary = [_rect(*r) for r in self._rects]
        self.centers = np.array([[(r[0] + r[2]) / 2, (r[1] + r[3]) / 2] for r in self._rects])
        self.centers = self.pixel2hand(self.centers)

    def pixel2hand(self, p):
        return (np.asarray(p, float) - _ORIGIN) / PPM

    def hand2pixel(self, h):
        return np.asarray(h, float) * PPM + _ORIGIN

    def locate(self, hand):
        px = self.hand2pixel(hand)
        ids = np.full(len(px), -1)
        for k, (x0, y0, x1, y1) in enumerate(self._rects):
            ids[(px[:, 0] >= x0) & (px[:, 0] < x1) & (px[:, 1] >= y0) & (px[:, 1] < y1)] = k
        return None, ids


class _Stim:
    def __init__(self, loc, radius):
        self.pins = [(np.asarray(loc, float), radius)]

    def __iadd__(self, other):
        self.pins += other.pins
        return self


class _Response:
    def __init__(self, spikes):
        self.spikes = spikes


class _Pop:
    def __init__(self, affclass, location):
        self.affclass, self.location = np.asarray(affclass), np.asarray(location)

    def __getitem__(self, mask):
        return _Pop(self.affclass[mask], self.location[mask])

    def __len__(self):
        return len(self.affclass)

    def __repr__(self):
        return f"FakePopulation with {len(self)} afferent(s)"

    def response(self, stim):
        rng = np.random.default_rng(0)
        pins = np.array([p[0] for p in stim.pins])
        d = np.linalg.norm(self.location[:, None, :] - pins[None], axis=2).min(axis=1)
        counts = rng.poisson(12 * np.exp(-d ** 2 / (2 * 8.0 ** 2)))
        return _Response([np.sort(rng.uniform(0, 3, c)) for c in counts])


class _Module:
    hand_surface = _HandSurface()
    last_pin_radius = None

    def stim_ramp(self, amp, len, loc, ramp_len, fs, pin_radius):
        self.last_pin_radius = pin_radius
        return _Stim(loc, pin_radius)

    def affpop_hand(self, density_multiplier=1.0):
        rng = np.random.default_rng(1)
        n = int(5000 * density_multiplier)
        cls = rng.choice(["SA1", "RA", "PC"], n, p=[.24, .55, .21])
        px = np.column_stack([rng.uniform(80, 380, n), rng.uniform(70, 460, n)])
        return _Pop(cls, self.hand_surface.pixel2hand(px))


def make_fake_touchsim():
    return _Module()
