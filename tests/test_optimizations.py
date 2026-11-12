import unittest
import json
import tempfile
from pathlib import Path
from itertools import combinations

import numba
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from multirocket.fast_ridge import FastRidgeClassifierCV
from multirocket.multirocket import MultiRocket, fit, transform, seed_numba
from multirocket.pooling import OPERATORS, PAIRS, feature_indices, transform_selected
from experiments.compare_optimizations import freeze_choice

numba.set_num_threads(2)


class ProtocolTests(unittest.TestCase):
    def test_complete_selection_required_and_choice_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'selection').mkdir()
            protocol = {'split': {'selection': ['A', 'B'], 'heldout': ['C']}}
            rows = [{'dataset': 'A', 'operators': '+'.join(pair), 'accuracy': .8}
                    for pair in PAIRS]
            (output / 'selection/A.json').write_text(json.dumps(rows))
            with self.assertRaises(FileNotFoundError):
                freeze_choice(output, protocol)
            (output / 'selection/B.json').write_text(json.dumps(rows[:3]))
            with self.assertRaises(RuntimeError):
                freeze_choice(output, protocol)
            (output / 'selection/B.json').write_text(json.dumps(rows))
            choice = freeze_choice(output, protocol)
            self.assertEqual(choice['operators'], list(PAIRS[0]))  # fixed tie rule
            # Held-out scores cannot influence selection.
            (output / 'selection/C.json').write_text('not a valid score file')
            self.assertEqual(choice, freeze_choice(output, protocol))
            rows[-1]['accuracy'] = 1
            (output / 'selection/B.json').write_text(json.dumps(rows))
            with self.assertRaises(RuntimeError):
                freeze_choice(output, protocol)


class PoolingTests(unittest.TestCase):
    def test_buffered_transform_preserves_original(self):
        from multirocket.buffered import transform_buffered
        for length in (10, 32, 99, 500):
            X = np.random.default_rng(length).normal(size=(7, length))
            X[0] = 0
            seed_numba(33)
            p, p1 = fit(X, 252), fit(np.diff(X), 252)
            assert_array_equal(transform_buffered(X, np.diff(X), p, p1, 4),
                               transform(X, np.diff(X), p, p1, 4))

    def test_fused_prediction_matches_scaled_pipeline(self):
        rng = np.random.default_rng(2026)
        X, Xt = rng.normal(size=(40, 50)), rng.normal(size=(19, 50))
        for classes in (1, 2, 4):
            y = np.arange(len(X)) % classes
            for backend in ('ridge', 'fast_ridge'):
                model = MultiRocket(2016, classifier=backend, random_state=3,
                                    memory_efficient=True, optimized_transform=True,
                                    fused_prediction=True, prediction_batch_size=7)
                model.fit(X, y, False)
                features = model._transform(Xt, np.diff(Xt))
                assert_array_equal(model._predict_features(features),
                                   model.classifier.predict(features.copy()))
                assert_array_equal(model.predict(Xt), model._predict_features(features))

    def test_selected_columns_match_original(self):
        for length in (10, 32, 99):
            X = np.random.default_rng(length).normal(size=(7, length))
            X[0] = 0
            seed_numba(33)
            p, p1 = fit(X, 252), fit(np.diff(X), 252)
            full = transform(X, np.diff(X), p, p1, 4)
            for pair in (*PAIRS, OPERATORS):
                actual = transform_selected(X, np.diff(X), p, p1, pair)
                assert_array_equal(actual, full[:, feature_indices(p, p1, pair)])

    def test_chunked_prediction_and_training_prediction(self):
        rng = np.random.default_rng(3)
        X, Xt = rng.normal(size=(32, 35)), rng.normal(size=(17, 35))
        y = np.arange(32) % 3
        for pair in (OPERATORS, PAIRS[2]):
            for backend in ('ridge', 'fast_ridge'):
                old = MultiRocket(2016, classifier=backend, pooling=pair, random_state=12)
                new = MultiRocket(2016, classifier=backend, pooling=pair, random_state=12,
                                  memory_efficient=True, prediction_batch_size=5)
                assert_array_equal(old.fit(X, y), new.fit(X, y))
                assert_array_equal(old.predict(Xt), new.predict(Xt))
                self.assertEqual(old.classifier[-1].alpha_, new.classifier[-1].alpha_)
                self.assertEqual(new.max_prediction_feature_bytes_, 5 * new.n_features_out_ * 4)
                self.assertFalse(any(isinstance(v, np.ndarray) and v.shape == (32, new.n_features_out_)
                                     for v in vars(new).values()))


class RidgeTests(unittest.TestCase):
    def test_against_explicit_leave_one_out(self):
        rng = np.random.default_rng(512)
        X = rng.normal(size=(15, 22)) + np.arange(22)
        y = np.arange(15) % 3
        alphas = np.array([0.01, 0.3, 10, 1000])
        target = 2 * (y[:, None] == np.arange(3)) - 1
        scores = []
        for alpha in alphas:
            predictions = []
            for i in range(len(X)):
                keep = np.arange(len(X)) != i
                fitted = RidgeClassifier(alpha=alpha).fit(X[keep], y[keep])
                predictions.append(fitted.decision_function(X[i:i+1])[0])
            scores.append(-np.mean((np.array(predictions) - target) ** 2))
        fast = FastRidgeClassifierCV(alphas=alphas, block_size=7).fit(X, y)
        assert_allclose(fast.loo_scores_, scores, rtol=1e-8, atol=1e-9)
        self.assertEqual(fast.alpha_, alphas[np.argmax(scores)])

    def test_alpha_scores_predictions_and_coefficients(self):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            for n, p in ((18, 72), (48, 20), (24, 24), (25, 150)):
                for dtype in (np.float32, np.float64):
                    X = rng.normal(size=(n, p)).astype(dtype)
                    X[:, 0] = 2
                    X[:, 1] = X[:, 2]  # rank deficient and nonzero means
                    if seed == 5:
                        X[:] = 0
                    y = np.array(['class_' + str(i % (2 + seed % 3)) for i in range(n)])
                    X = StandardScaler().fit_transform(X)
                    alphas = np.logspace(-3, 3, 10)
                    old = RidgeClassifierCV(alphas=alphas).fit(X, y)
                    new = FastRidgeClassifierCV(alphas=alphas, block_size=11).fit(X, y)
                    self.assertEqual(old.alpha_, new.alpha_, (seed, n, p, dtype))
                    assert_allclose(old.best_score_, new.best_score_, rtol=1e-7, atol=1e-8)
                    assert_allclose(np.atleast_2d(old.coef_), np.atleast_2d(new.coef_), rtol=1e-4, atol=1e-6)
                    Xt = rng.normal(size=(15, p)).astype(dtype)
                    assert_array_equal(old.predict(Xt), new.predict(Xt))

    def test_invalid_options(self):
        X = np.ones((4, 6))
        for alphas in ([], [0], [-1], [float('nan')]):
            with self.assertRaises(ValueError):
                FastRidgeClassifierCV(alphas).fit(X, [0, 0, 1, 1])
        for pooling in ([], ['PPV', 'PPV'], ['oops']):
            with self.assertRaises(ValueError):
                MultiRocket(pooling=pooling)


if __name__ == '__main__':
    unittest.main()
