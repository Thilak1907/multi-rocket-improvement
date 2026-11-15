import unittest
import pickle
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from multirocket.feature_power import FeaturePowerMultiRocket, map_features_inplace
from multirocket.presets import make_multirocket
from multirocket.accuracy_ensemble import ridge_scores, labels_from_scores


class FeaturePowerTests(unittest.TestCase):
    def test_fixed_maps_preserve_sign_and_mutate_existing_buffer(self):
        values=np.array([[-4.,-1.,0.,1.,9.]],dtype=np.float32)
        for logarithmic in (False,True):
            data=values.copy()
            expected=np.sign(values)*(np.log1p(np.abs(values)) if logarithmic else np.sqrt(np.abs(values)))
            result=map_features_inplace(data,logarithmic)
            self.assertTrue(np.shares_memory(data,result))
            assert_allclose(result,expected,atol=1e-6)

    def test_same_kernels_transformed_columns_and_chunked_serialization(self):
        rng=np.random.default_rng(14)
        X,Xt,y=rng.normal(size=(18,29)),rng.normal(size=(7,29)),np.arange(18)%3
        base=make_multirocket(random_state=14)
        base.fit(X,y,predict_on_train=False)
        raw=base._transform(Xt,np.diff(Xt,axis=1))
        for name in ('sqrt','log'):
            model=FeaturePowerMultiRocket(feature_map=name,random_state=14,prediction_batch_size=2)
            model.fit(X,y,predict_on_train=False)
            assert_array_equal(model.base_parameters[2],base.base_parameters[2])
            assert_array_equal(model.diff1_parameters[2],base.diff1_parameters[2])
            expected=np.sign(raw)*(np.log1p(np.abs(raw)) if name=='log' else np.sqrt(np.abs(raw)))
            assert_allclose(model._transform(Xt,np.diff(Xt,axis=1)),expected,rtol=1e-6,atol=1e-6)
            pred=model.predict(Xt)
            assert_array_equal(pred,labels_from_scores(ridge_scores(model,Xt),model.classifier[-1].classes_))
            restored=pickle.loads(pickle.dumps(model))
            assert_array_equal(restored.predict(Xt),pred)
            self.assertEqual(model.n_features_out_,29568)
