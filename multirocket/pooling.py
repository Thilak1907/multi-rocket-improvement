"""Selected pooling without allocating the discarded feature columns.

Operator definitions (including boundary/equality conventions) match this repo's
original transform. The four-operator transform remains the reference path.
"""
from itertools import combinations

import numpy as np
from numba import njit, prange

OPERATORS = ("PPV", "LSPV", "MPV", "MIPV")
PAIRS = tuple(combinations(OPERATORS, 2))
_INDICES = np.asarray(list(combinations(range(9), 3)), dtype=np.int32)


def validate_pooling(pooling):
    if isinstance(pooling, str):
        pooling = pooling.upper().replace(",", "+").split("+")
    pooling = tuple(pooling)
    if not pooling or len(set(pooling)) != len(pooling) or any(p not in OPERATORS for p in pooling):
        raise ValueError("pooling must contain distinct operators from " + str(OPERATORS))
    return tuple(p for p in OPERATORS if p in pooling)


def feature_indices(parameters, parameters1, pooling):
    """Columns in the original transform; useful for an independent oracle."""
    pooling = validate_pooling(pooling)
    n, n1 = len(parameters[2]), len(parameters1[2])
    return np.concatenate([
        np.arange(offset + OPERATORS.index(p) * width,
                  offset + (OPERATORS.index(p) + 1) * width)
        for offset, width in ((0, n), (4 * n, n1)) for p in pooling
    ])


@njit(fastmath=True, cache=True)
def _pool(C, bias, slots, features, row, col, width):
    ppv = 0
    last_val = 0
    max_stretch = 0.0
    mean_index = 0
    mean = 0.0
    for j in range(len(C)):
        if C[j] > bias:
            ppv += 1
            if slots[2] >= 0:
                mean += C[j] + bias
            if slots[3] >= 0:
                mean_index += j
        elif slots[1] >= 0 and C[j] < bias:
            stretch = j - last_val
            if stretch > max_stretch:
                max_stretch = stretch
            last_val = j
    if slots[0] >= 0:
        features[row, col + slots[0] * width] = ppv / len(C)
    if slots[1] >= 0:
        stretch = len(C) - 1 - last_val
        if stretch > max_stretch:
            max_stretch = stretch
        features[row, col + slots[1] * width] = max_stretch
    if slots[2] >= 0:
        features[row, col + slots[2] * width] = mean / ppv if ppv > 0 else 0
    if slots[3] >= 0:
        features[row, col + slots[3] * width] = mean_index / ppv if ppv > 0 else -1


@njit(fastmath=True, parallel=True, cache=True)
def _representation(X, parameters, slots, features, offset, difference):
    dilations, counts, biases = parameters
    width = len(biases)
    length = X.shape[1]
    for row in prange(len(X)):
        A = -X[row]
        G = X[row] + X[row] + X[row]
        feature_start = 0
        for di in range(len(dilations)):
            dilation = dilations[di]
            padding = 4 * dilation
            C_alpha = np.zeros(length, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, length), dtype=np.float32)
            C_gamma[4] = G
            start = dilation
            # Preserve the original first-difference alignment exactly.
            end = length + difference - padding
            for gi in range(4):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gi, -end:] = G[:end]
                end += dilation
            for gi in range(5, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gi, :-start] = G[start:]
                start += dilation
            for ki in range(84):
                i, j, k = _INDICES[ki]
                C = C_alpha + C_gamma[i] + C_gamma[j] + C_gamma[k]
                if (di + ki) % 2:
                    C = C[padding:-padding]
                for fi in range(counts[di]):
                    col = feature_start + fi
                    _pool(C, biases[col], slots, features, row, offset + col, width)
                feature_start += counts[di]


def transform_selected(X, X1, parameters, parameters1, pooling, first_difference_fix=False):
    pooling = validate_pooling(pooling)
    slots = np.asarray([pooling.index(p) if p in pooling else -1 for p in OPERATORS], dtype=np.int32)
    offset = len(parameters[2]) * len(pooling)
    features = np.empty((len(X), offset + len(parameters1[2]) * len(pooling)), dtype=np.float32)
    _representation(X, parameters, slots, features, 0, 0)
    _representation(X1, parameters1, slots, features, offset, 0 if first_difference_fix else 1)
    return features
