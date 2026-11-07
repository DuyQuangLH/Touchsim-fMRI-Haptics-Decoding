"""Spike-count features and similarity between responses."""
import numpy as np


def spike_counts(resp) -> np.ndarray:
    """Number of spikes of every afferent."""
    return np.array([len(s) for s in resp.spikes])


def evaluate(counts_a, counts_b) -> float:
    """Pearson r between two spike-count vectors (0.0 if either is constant)."""
    if np.std(counts_a) == 0 or np.std(counts_b) == 0:
        return 0.0
    return float(np.corrcoef(counts_a, counts_b)[0, 1])


def similarity_matrix(all_counts: dict) -> np.ndarray:
    """(n, n) Pearson r between the responses of every pair of digits."""
    keys = sorted(all_counts)
    return np.array([[evaluate(all_counts[a], all_counts[b]) for b in keys] for a in keys])


def counts_by_class(counts, pop) -> dict:
    """Total spikes per afferent class."""
    cls = np.array(pop.affclass)
    return {k: int(counts[cls == k].sum()) for k in ("SA1", "RA", "PC")}
