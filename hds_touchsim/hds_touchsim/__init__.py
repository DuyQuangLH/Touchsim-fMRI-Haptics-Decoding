from .hds import HDSSpec, digit_tensor, fit_to_region, hamming_matrix
from .adapter import (CLASS_RATIO, PALM_REGIONS, REAL_HAND_LENGTH_MM, balance_classes,
                      build_stimulus, hand_scale, load_touchsim, make_population, px_per_mm,
                      region_mask_fn, respond)
from .evaluate import counts_by_class, evaluate, similarity_matrix, spike_counts
