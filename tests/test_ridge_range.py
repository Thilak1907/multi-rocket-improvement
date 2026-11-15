import unittest
import numpy as np
from numpy.testing import assert_array_equal
from multirocket.presets import make_multirocket


class RidgeRangeTests(unittest.TestCase):
    def test_default_grid_preserved_and_expanded_grid_matches_sklearn(self):
        rng=np.random.default_rng(12)
        X,Xt,y=rng.normal(size=(24,27)),rng.normal(size=(9,27)),np.arange(24)%3
        default=np.logspace(-3,3,10)
        expanded=np.r_[np.logspace(-6,-4,3),default,np.logspace(4,6,3)]
        a=make_multirocket(random_state=12)
        b=make_multirocket(random_state=12,ridge_alphas=default)
        a.fit(X,y,predict_on_train=False);b.fit(X,y,predict_on_train=False)
        assert_array_equal(a.predict(Xt),b.predict(Xt))
        self.assertEqual(a.classifier[-1].alpha_,b.classifier[-1].alpha_)
        c=make_multirocket(random_state=12,ridge_alphas=expanded)
        d=make_multirocket(random_state=12,ridge_alphas=expanded,classifier='ridge')
        c.fit(X,y,predict_on_train=False);d.fit(X,y,predict_on_train=False)
        assert_array_equal(c.predict(Xt),d.predict(Xt))
        self.assertEqual(c.classifier[-1].alpha_,d.classifier[-1].alpha_)
        self.assertEqual(c.n_features_out_,29568)
