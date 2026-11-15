"""Experimental fixed nonlinear feature maps before the retained ridge pipeline.

These maps keep 29,568 columns; they change the representation and may change
accuracy. No fitted distribution parameters or test-set statistics are used.
"""
import numpy as np
from numba import njit, prange
from multirocket.multirocket import MultiRocket
from multirocket.presets import PRESETS


@njit(cache=True, parallel=True)
def map_features_inplace(features, logarithmic):
    for row in prange(features.shape[0]):
        for col in range(features.shape[1]):
            value = features[row, col]
            magnitude = np.log1p(abs(value)) if logarithmic else np.sqrt(abs(value))
            features[row, col] = -magnitude if value < 0 else magnitude
    return features


class FeaturePowerMultiRocket(MultiRocket):
    def __init__(self, feature_map='sqrt', **options):
        if feature_map not in ('sqrt', 'log'):
            raise ValueError('feature_map must be sqrt or log')
        self.feature_map = feature_map
        super().__init__(**{**PRESETS['previous30'], **options})

    def _transform(self, X, X1):
        features = super()._transform(X, X1)
        np.nan_to_num(features, copy=False)
        return map_features_inplace(features, self.feature_map == 'log')
