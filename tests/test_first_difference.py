"""Independent convolution oracle for the L-1 difference alignment."""
import unittest
from itertools import combinations
import numpy as np
from numpy.testing import assert_array_equal
from multirocket.multirocket import MultiRocket, transform
from multirocket.corrected import transform_corrected, transform_buffered_corrected
from multirocket.pooling import OPERATORS, PAIRS, transform_selected, feature_indices


def direct_features(X, parameters):
    dilations, counts, biases = parameters
    width = len(biases)
    result = np.empty((len(X),4*width),dtype=np.float32)
    for row,x in enumerate(X):
        column=0
        for di,d in enumerate(dilations):
            padding=4*int(d)
            for ki,taps in enumerate(combinations(range(9),3)):
                weights=np.full(9,-1.)
                weights[list(taps)]+=3
                kernel=np.zeros(8*int(d)+1)
                kernel[::int(d)]=weights
                response=np.correlate(np.pad(x,(padding,padding)),kernel,mode='valid')
                if (di+ki)%2: response=response[padding:-padding]
                for _ in range(int(counts[di])):
                    bias=biases[column]
                    positive=response>bias
                    indices=np.flatnonzero(positive)
                    # Preserve this repo's existing LSPV boundary convention.
                    negative=np.flatnonzero(response<bias)
                    stretch=np.diff(np.r_[0,negative,len(response)-1]).max(initial=0)
                    values=(positive.mean(),stretch,
                            (response[positive]+bias).mean() if len(indices) else 0,
                            indices.mean() if len(indices) else -1)
                    for operator,value in enumerate(values): result[row,column+operator*width]=value
                    column+=1
    return result


class DifferenceTests(unittest.TestCase):
    def test_presets_keep_legacy_and_fixed_backend_available(self):
        from multirocket.presets import make_multirocket
        previous=make_multirocket('previous30')
        candidate=make_multirocket('candidate30')
        legacy=make_multirocket('legacy30')
        self.assertEqual(previous.clf,'fast_ridge')
        self.assertEqual(candidate.clf,'auto_ridge')
        self.assertTrue(previous.first_difference_fix)
        self.assertTrue(candidate.first_difference_fix)
        self.assertFalse(legacy.first_difference_fix)
        self.assertEqual(candidate.ridge_feature_limit_bytes,693*1024**2)
        self.assertEqual(candidate.feature_budget,30000)

    def test_buffered_corrected_float_features_are_identical(self):
        from multirocket.multirocket import fit,seed_numba
        X=np.random.default_rng(997).normal(size=(5,99))
        seed_numba(77)
        p,p1=fit(X,6250),fit(np.diff(X),6250)
        assert_array_equal(transform_corrected(X,np.diff(X),p,p1,4),
                           transform_buffered_corrected(X,np.diff(X),p,p1,4))

    def test_corrected_transforms_match_direct_convolution(self):
        for length in (10,33,99):
            X=np.random.default_rng(length).integers(-3,4,size=(4,length)).astype(np.float64)
            X[0]=0
            d=np.array([v for v in (1,2,4) if 8*v<length-1],dtype=np.int32)
            count=np.ones(len(d),dtype=np.int32)
            width=84*len(d)
            params=(d,count,(np.arange(width)%5+.25).astype(np.float32))
            diff=np.diff(X)
            expected=np.concatenate((direct_features(X,params),direct_features(diff,params)),axis=1)
            fixed=transform_corrected(X,diff,params,params,4)
            assert_array_equal(fixed,expected)
            assert_array_equal(transform_buffered_corrected(X,diff,params,params,4),expected)
            for pair in (*PAIRS,OPERATORS):
                assert_array_equal(transform_selected(X,diff,params,params,pair,True),
                                   expected[:,feature_indices(params,params,pair)])
            legacy=transform(X,diff,params,params,4)
            assert_array_equal(legacy[:,:4*width],fixed[:,:4*width])
            self.assertGreater(np.count_nonzero(legacy[:,4*width:]!=fixed[:,4*width:]),0)

    def test_auto_ridge_byte_threshold_and_independent_memory_controls(self):
        rng=np.random.default_rng(83)
        X=rng.normal(size=(24,35)); Xt=rng.normal(size=(11,35)); y=np.arange(24)%3
        size=24*2016*4
        for limit,expected in ((size+1,'ridge'),(size,'fast_ridge'),(size-1,'fast_ridge')):
            model=MultiRocket(2016,classifier='auto_ridge',random_state=3,
                              ridge_feature_limit_bytes=limit)
            model.fit(X,y,False)
            self.assertEqual(model.selected_ridge_backend_,expected)
            self.assertEqual(model.estimated_train_feature_bytes_,size)
            reference=MultiRocket(2016,classifier=expected,random_state=3)
            reference.fit(X,y,False)
            assert_array_equal(model.predict(Xt),reference.predict(Xt))
        reference=MultiRocket(2016,random_state=3)
        reference.fit(X,y,False)
        for inplace,chunked in ((True,False),(False,True)):
            model=MultiRocket(2016,random_state=3,inplace_processing=inplace,
                              chunked_prediction=chunked,prediction_batch_size=4)
            assert_array_equal(model.fit(X,y),reference.classifier.predict(reference._transform(X,np.diff(X))))
            assert_array_equal(model.predict(Xt),reference.predict(Xt))
            self.assertEqual(model.max_prediction_feature_bytes_,(4 if chunked else 11)*2016*4)


if __name__=='__main__': unittest.main()
