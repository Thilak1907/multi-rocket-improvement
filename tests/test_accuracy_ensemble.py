import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from multirocket.accuracy_ensemble import (
    MultiRocketAccuracyEnsemble, labels_from_scores, ridge_scores, series_view,
)
from multirocket.presets import make_multirocket


class AccuracyEnsembleTests(unittest.TestCase):
    def test_views_handle_constants_and_short_series(self):
        X = np.array([[1.] * 20, list(range(20))])
        normalized = series_view(X, 'normalized')
        assert_array_equal(normalized[0], np.zeros(20))
        assert_allclose(normalized[1].mean(), 0., atol=1e-12)
        assert_allclose(normalized[1].std(), 1.)
        assert_array_equal(series_view(X, 'coarse')[1], np.arange(.5, 20, 2))
        assert_array_equal(series_view(X[:, :11], 'coarse'), X[:, :11])
        assert_array_equal(X[1], np.arange(20))

    def test_scores_match_recommended_prediction_and_chunking(self):
        rng = np.random.default_rng(5)
        X, Xt = rng.normal(size=(18, 25)), rng.normal(size=(9, 25))
        for y in (np.arange(18) % 2, np.arange(18) % 3):
            model = make_multirocket(random_state=5, prediction_batch_size=2)
            model.fit(X, y, predict_on_train=False)
            scores = ridge_scores(model, Xt)
            assert_array_equal(labels_from_scores(scores, model.classifier[-1].classes_), model.predict(Xt))
            model.prediction_batch_size = 20
            assert_allclose(scores, ridge_scores(model, Xt), atol=1e-10)

    def test_ensemble_is_exact_member_average_and_roundtrips(self):
        import pickle
        rng = np.random.default_rng(9)
        X, Xt, y = rng.normal(size=(16, 23)), rng.normal(size=(4, 23)), np.arange(16) % 2
        model = MultiRocketAccuracyEnsemble(view='coarse', random_state=9).fit(X, y)
        expected = (ridge_scores(model.members_[0], Xt) + ridge_scores(model.members_[1], series_view(Xt, 'coarse'))) / 2
        assert_allclose(model.decision_function(Xt), expected)
        restored = pickle.loads(pickle.dumps(model))
        assert_array_equal(restored.predict(Xt), model.predict(Xt))
        self.assertEqual(model.members_[0].n_features_out_, 29568)
        self.assertEqual(model.members_[1].n_features_out_, 29568)
