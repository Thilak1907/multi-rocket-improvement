"""CPU HYDRA / MultiRocket+HYDRA benchmark adapters.

HYDRA and its sparse scaler are the unmodified upstream implementation in
vendor/hydra.py (GPL-3.0). MultiRocket columns use StandardScaler independently;
HYDRA columns use the paper's SparseScaler, before a shared ridge classifier.
"""
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

from experiments.vendor.hydra import Hydra, SparseScaler
from multirocket.multirocket import fit, seed_numba
from multirocket.buffered import transform_buffered


class HydraRocket:
    def __init__(self, budget=0, groups=64, seed=2026, batch_size=64, first_difference_fix=True):
        self.budget, self.groups, self.seed = budget, groups, seed
        self.batch_size = batch_size
        self.first_difference_fix = first_difference_fix

    def _features(self, X):
        with torch.inference_mode():
            tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)[:, None, :].copy())
            hydro = self.hydra.batch(tensor, self.batch_size)
        if self.budget:
            if self.first_difference_fix:
                from multirocket.corrected import transform_buffered_corrected
                rocket = transform_buffered_corrected(X, np.diff(X), self.p, self.p1, 4)
            else:
                rocket = transform_buffered(X, np.diff(X), self.p, self.p1, 4)
            np.nan_to_num(rocket, copy=False)
        else:
            rocket = None
        return rocket, hydro

    def fit(self, X, y, predict_on_train=False):
        X = np.asarray(X, dtype=np.float64)
        # Bound conv1d scratch to about eight million float32 activations.
        self.batch_size = max(1, min(self.batch_size, 8_000_000 // (X.shape[1] * self.groups * 4)))
        self.hydra = Hydra(X.shape[1], g=self.groups, seed=self.seed)
        self.hs = SparseScaler()
        if self.budget:
            seed_numba(self.seed)
            self.p, self.p1 = fit(X, self.budget // 8), fit(np.diff(X), self.budget // 8)
        t = time.perf_counter()
        rocket, hydro = self._features(X)
        self.apply_kernel_on_train_duration = time.perf_counter() - t
        t = time.perf_counter()
        with torch.inference_mode():
            hydro = self.hs.fit_transform(hydro).numpy()
        if self.budget:
            self.rs = StandardScaler(copy=False)
            self.rs.fit_transform(rocket)
            features = np.concatenate((rocket, hydro), axis=1)
            del rocket, hydro
        else:
            features = hydro
        self.n_features_out_ = features.shape[1]
        self.train_feature_bytes_ = features.nbytes
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(features, y)
        self.train_duration = time.perf_counter() - t
        self.alpha_ = float(self.classifier.alpha_)
        del features
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = []
        self.apply_kernel_on_test_duration = self.test_duration = 0.0
        self.max_prediction_feature_bytes_ = 0
        for start in range(0, len(X), self.batch_size):
            t = time.perf_counter()
            rocket, hydro = self._features(X[start:start+self.batch_size])
            self.apply_kernel_on_test_duration += time.perf_counter() - t
            t = time.perf_counter()
            with torch.inference_mode():
                hydro = self.hs.transform(hydro).numpy()
            if self.budget:
                self.rs.transform(rocket)
                features = np.concatenate((rocket, hydro), axis=1)
                del rocket, hydro
            else:
                features = hydro
            self.max_prediction_feature_bytes_ = max(self.max_prediction_feature_bytes_, features.nbytes)
            out.append(self.classifier.predict(features))
            self.test_duration += time.perf_counter() - t
            del features
        return np.concatenate(out)
