# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
# https://arxiv.org/abs/2102.00457

import time

import numpy as np
from numba import njit, prange
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from multirocket.logistic_regression import LogisticRegression
from multirocket.fast_ridge import FastRidgeClassifierCV
from multirocket.pooling import OPERATORS, validate_pooling, transform_selected


@njit(cache=True)
def seed_numba(seed):
    """Numba has a separate RNG; seed it for paired, reproducible comparisons."""
    np.random.seed(seed)


@njit("float32[:](float64[:,:],int32[:],int32[:],float32[:])",
      fastmath=True, parallel=False, cache=True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles):
    num_examples, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]

            A = -_X  # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[
                                                                           feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

    return biases


def _fit_dilations(input_length, num_features, max_dilations_per_kernel):
    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)  # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


def fit(X, num_features=10_000, max_dilations_per_kernel=32):
    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)

    return dilations, num_features_per_dilation, biases


@njit(
    "float32[:,:](float64[:,:],float64[:,:],Tuple((int32[:],int32[:],float32[:])),Tuple((int32[:],int32[:],float32[:])),int32)",
    fastmath=True, parallel=True, cache=True)
def transform(X, X1, parameters, parameters1, n_features_per_kernel=4):
    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters
    dilations1, num_features_per_dilation1, biases1 = parameters1

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_dilations1 = len(dilations1)

    num_features = num_kernels * np.sum(num_features_per_dilation)
    num_features1 = num_kernels * np.sum(num_features_per_dilation1)

    features = np.zeros((num_examples, (num_features + num_features1) * n_features_per_kernel), dtype=np.float32)
    n_features_per_transform = np.int64(features.shape[1] / 2)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + \
                    C_gamma[index_0] + \
                    C_gamma[index_1] + \
                    C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1

                feature_index_start = feature_index_end

        # First order difference
        _X1 = X1[example_index]
        A1 = -_X1  # A = alpha * X = -X
        G1 = _X1 + _X1 + _X1  # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations1):

            _padding0 = dilation_index % 2

            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = np.zeros(input_length - 1, dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, input_length - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A1[:end]
                C_gamma[gamma_index, -end:] = G1[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A1[start:]
                C_gamma[gamma_index, :-start] = G1[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + \
                    C_gamma[index_0] + \
                    C_gamma[index_1] + \
                    C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                else:
                    _c = C[padding:-padding]

                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]

                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0

                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index + n_features_per_transform
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features
                        features[example_index, end] = max_stretch
                        end = end + num_features
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1

                feature_index_start = feature_index_end

    return features


class MultiRocket:

    def __init__(
            self,
            num_features=50000,
            classifier="ridge",
            verbose=0,
            pooling=OPERATORS,
            random_state=None,
            memory_efficient=False,
            prediction_batch_size=256,
            optimized_transform=False,
            fused_prediction=False,
            first_difference_fix=True,
            inplace_processing=None,
            chunked_prediction=None,
            ridge_feature_limit_bytes=693 * 1024**2,
    ):
        self.name = "MultiRocket"

        self.base_parameters = None
        self.diff1_parameters = None

        self.pooling = validate_pooling(pooling)
        self.n_features_per_kernel = len(self.pooling)
        self.feature_budget = int(num_features)
        self.num_features = num_features / 2  # 1 per transformation
        # Keep the original kernel budget: selecting two operators halves features.
        self.num_kernels = int(self.num_features / 4)
        if self.num_kernels < 84:
            raise ValueError("num_features must be at least 672")
        if classifier.lower() not in ("ridge", "fast_ridge", "auto_ridge", "logistic"):
            raise ValueError("classifier must be ridge, fast_ridge, auto_ridge, or logistic")
        if not isinstance(prediction_batch_size, (int, np.integer)) or prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be a positive integer")
        self.random_state = random_state
        self.memory_efficient = memory_efficient
        self.inplace_processing = memory_efficient if inplace_processing is None else inplace_processing
        self.chunked_prediction = memory_efficient if chunked_prediction is None else chunked_prediction
        self.first_difference_fix = first_difference_fix
        if not isinstance(ridge_feature_limit_bytes, (int, np.integer)) or ridge_feature_limit_bytes < 1:
            raise ValueError("ridge_feature_limit_bytes must be a positive integer")
        self.ridge_feature_limit_bytes = ridge_feature_limit_bytes
        self.prediction_batch_size = prediction_batch_size
        self.optimized_transform = optimized_transform
        self.fused_prediction = fused_prediction
        if (optimized_transform and self.pooling != OPERATORS):
            raise ValueError("optimized_transform currently requires all four operators")
        if fused_prediction and classifier.lower() == "logistic":
            raise ValueError("fused_prediction requires a ridge classifier")

        if verbose > 1:
            print('[{}] Creating {} with {} kernels'.format(self.name, self.name, self.num_kernels))

        self.clf = classifier
        self.classifier = None
        self.train_duration = 0
        self.test_duration = 0
        self.generate_kernel_duration = 0
        self.train_transforms_duration = 0
        self.test_transforms_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.apply_kernel_on_test_duration = 0

        self.verbose = verbose

    def fit(self, x_train, y_train, predict_on_train=True):
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim != 2 or x_train.shape[1] < 10:
            raise ValueError("Expected a 2D array of series with length at least 10")
        if self.random_state is not None:
            seed_numba(self.random_state)
        if self.verbose > 1:
            print('[{}] Training with training set of {}'.format(self.name, x_train.shape))

        self.generate_kernel_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.train_transforms_duration = 0

        start_time = time.perf_counter()

        _start_time = time.perf_counter()
        xx = np.diff(x_train, 1)
        self.train_transforms_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        self.base_parameters = fit(
            x_train,
            num_features=self.num_kernels
        )
        self.diff1_parameters = fit(
            xx,
            num_features=self.num_kernels
        )
        self.generate_kernel_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        x_train_transform = self._transform(
            x_train, xx,
        )
        del xx
        self.apply_kernel_on_train_duration += time.perf_counter() - _start_time

        x_train_transform = np.nan_to_num(x_train_transform, copy=not self.inplace_processing)
        self.n_features_out_ = x_train_transform.shape[1]
        self.train_feature_bytes_ = x_train_transform.nbytes
        self.estimated_train_feature_bytes_ = len(x_train) * self.n_features_out_ * np.dtype(np.float32).itemsize
        self.selected_ridge_backend_ = self.clf.lower()
        if self.selected_ridge_backend_ == "auto_ridge":
            self.selected_ridge_backend_ = ("fast_ridge" if self.estimated_train_feature_bytes_ >= self.ridge_feature_limit_bytes else "ridge")

        elapsed_time = time.perf_counter() - start_time
        if self.verbose > 1:
            print('[{}] Kernels applied!, took {}s'.format(self.name, elapsed_time))
            print('[{}] Transformed Shape {}'.format(self.name, x_train_transform.shape))

        if self.verbose > 1:
            print('[{}] Training'.format(self.name))

        if self.selected_ridge_backend_ in ("ridge", "fast_ridge"):
            ridge = FastRidgeClassifierCV if self.selected_ridge_backend_ == "fast_ridge" else RidgeClassifierCV
            self.classifier = make_pipeline(
                StandardScaler(copy=not self.inplace_processing),
                ridge(
                    alphas=np.logspace(-3, 3, 10)
                )
            )
        else:
            self.classifier = LogisticRegression(
                num_features=x_train_transform.shape[1],
                max_epochs=200,
            )
        _start_time = time.perf_counter()
        self.classifier.fit(x_train_transform, y_train)
        self.train_duration = time.perf_counter() - _start_time

        if self.fused_prediction:
            scaler, fitted = self.classifier[0], self.classifier[-1]
            self.prediction_coef_ = np.atleast_2d(fitted.coef_).astype(np.float64) / scaler.scale_
            self.prediction_intercept_ = fitted.intercept_ - self.prediction_coef_ @ scaler.mean_

        if self.verbose > 1:
            print('[{}] Training done!, took {:.3f}s'.format(self.name, self.train_duration))
        if predict_on_train:
            if self.inplace_processing and self.selected_ridge_backend_ in ("ridge", "fast_ridge"):
                # fit already scaled this array in place; do not scale it twice.
                yhat = self.classifier[-1].predict(x_train_transform)
            else:
                yhat = self.classifier.predict(x_train_transform)
        else:
            yhat = None

        del x_train_transform
        self.fit_total_duration = time.perf_counter() - start_time
        return yhat

    def _transform(self, x, xx):
        if self.first_difference_fix and self.pooling == OPERATORS:
            from multirocket.corrected import transform_corrected, transform_buffered_corrected
            function = transform_buffered_corrected if self.optimized_transform else transform_corrected
            return function(x, xx, self.base_parameters, self.diff1_parameters, 4)
        if self.optimized_transform:
            from multirocket.buffered import transform_buffered
            return transform_buffered(x, xx, self.base_parameters, self.diff1_parameters, 4)
        if self.pooling == OPERATORS:
            return transform(x, xx, self.base_parameters, self.diff1_parameters, 4)
        return transform_selected(x, xx, self.base_parameters, self.diff1_parameters, self.pooling,
                                  first_difference_fix=self.first_difference_fix)

    def _predict_features(self, features):
        if not self.fused_prediction:
            return self.classifier.predict(features)
        fitted = self.classifier[-1]
        scores = features @ self.prediction_coef_.T + self.prediction_intercept_
        if len(fitted.classes_) == 1:
            return np.full(len(features), fitted.classes_[0])
        indices = (scores[:, 0] > 0).astype(int) if scores.shape[1] == 1 else scores.argmax(axis=1)
        return fitted.classes_[indices]

    def predict(self, x):
        if self.classifier is None:
            raise ValueError("Call fit before predict")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("Expected a 2D array")
        if self.chunked_prediction:
            start = time.perf_counter()
            self.apply_kernel_on_test_duration = 0
            self.test_transforms_duration = 0
            self.test_duration = 0
            self.max_prediction_feature_bytes_ = 0
            predictions = []
            for first in range(0, len(x), self.prediction_batch_size):
                chunk = x[first:first + self.prediction_batch_size]
                t = time.perf_counter()
                xx = np.diff(chunk, 1)
                self.test_transforms_duration += time.perf_counter() - t
                t = time.perf_counter()
                features = self._transform(chunk, xx)
                del xx
                self.apply_kernel_on_test_duration += time.perf_counter() - t
                features = np.nan_to_num(features, copy=not self.inplace_processing)
                self.max_prediction_feature_bytes_ = max(self.max_prediction_feature_bytes_, features.nbytes)
                t = time.perf_counter()
                predictions.append(self._predict_features(features))
                self.test_duration += time.perf_counter() - t
                del features
            self.predict_total_duration = time.perf_counter() - start
            if not predictions:
                return np.empty(0, dtype=self.classifier.classes_.dtype)
            return np.concatenate(predictions)

        total_start = time.perf_counter()
        if self.verbose > 1:
            print('[{}] Predicting'.format(self.name))

        self.apply_kernel_on_test_duration = 0
        self.test_transforms_duration = 0

        _start_time = time.perf_counter()
        xx = np.diff(x, 1)
        self.test_transforms_duration += time.perf_counter() - _start_time

        _start_time = time.perf_counter()
        x_transform = self._transform(
            x, xx,
        )
        del xx
        self.apply_kernel_on_test_duration += time.perf_counter() - _start_time

        x_transform = np.nan_to_num(x_transform, copy=not self.inplace_processing)
        self.max_prediction_feature_bytes_ = x_transform.nbytes
        if self.verbose > 1:
            print('Kernels applied!, took {:.3f}s. Transformed shape: {}.'.format(self.apply_kernel_on_test_duration,
                                                                                  x_transform.shape))

        start_time = time.perf_counter()
        yhat = self._predict_features(x_transform)
        self.test_duration = time.perf_counter() - start_time
        if self.verbose > 1:
            print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        del x_transform
        self.predict_total_duration = time.perf_counter() - total_start
        return yhat
