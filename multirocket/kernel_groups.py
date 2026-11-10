from itertools import combinations

import numpy as np


NUM_KERNELS = 84


def validate_random_state(random_state):
    if random_state is None:
        return None
    if (isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
            or random_state < 0
            or random_state > np.iinfo(np.int32).max):
        raise ValueError("random_state must be an integer from 0 to 2**31 - 1 or None")
    return int(random_state)


def select_kernel_indices(grouping="none"):
    if grouping == "none":
        return np.arange(NUM_KERNELS, dtype=np.int32)
    if grouping != "temporal_scale":
        raise ValueError(
            "kernel_grouping must be 'none' or 'temporal_scale'"
        )

    patterns = np.asarray(list(combinations(range(9), 3)), dtype=np.int32)
    spans = patterns[:, 2] - patterns[:, 0]
    groups = (spans <= 3, (spans >= 4) & (spans <= 5), spans >= 6)

    selected = []
    for group_mask in groups:
        group_indices = np.flatnonzero(group_mask)
        count = (len(group_indices) + 1) // 2
        positions = np.linspace(0, len(group_indices) - 1, count, dtype=np.int32)
        selected.extend(group_indices[positions])

    return np.asarray(sorted(selected), dtype=np.int32)
