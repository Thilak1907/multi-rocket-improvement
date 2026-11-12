import unittest
import numpy as np
from experiments.summarize_revision import paired_ni,aggregate_accuracy,accuracy_inference
from experiments.review_revision import capped_indices


class ReviewStatisticsTests(unittest.TestCase):
    def test_resample_column_and_timing_repeats_aggregate_correctly(self):
        import pandas as pd
        rows=[];names=[f'D{i}' for i in range(5)]
        for dataset in names:
            for split in (1,2,3):
                for variant,delta in [('baseline50',0),('auto30',.001)]:
                    for repeat in range(3 if split==1 else 1):
                        value=.7+split*.01+delta
                        rows.append(dict(dataset=dataset,resample=split,variant=variant,repeat=repeat,
                                         status='ok',accuracy=value,balanced_accuracy=value,macro_f1=value))
        acc=aggregate_accuracy(pd.DataFrame(rows))
        self.assertEqual(len(acc),5*3*2)
        ni,deltas=accuracy_inference(acc,('auto30',),names,[1,2,3],-.3)
        self.assertEqual(int(ni.iloc[0].datasets),5)
        self.assertEqual(int(ni.iloc[0].resamples),3)
        self.assertAlmostEqual(ni.iloc[0].mean_delta_pp,.1)
        self.assertTrue(ni.iloc[0].noninferior)
        self.assertEqual(len(deltas),5)

    def test_noninferiority_uses_datasets_and_margin(self):
        unchanged=np.zeros((55,3))
        result=paired_ni(unchanged,-.3)
        self.assertTrue(result['noninferior'])
        self.assertEqual(result['datasets'],55)
        self.assertEqual(result['resamples'],3)
        self.assertEqual(result['ci_low_pp'],0)
        self.assertFalse(paired_ni(np.full((55,3),-.4),-.3)['noninferior'])
        self.assertFalse(paired_ni(np.full((55,3),-.3),-.3)['noninferior'])

    def test_repeating_a_timing_measurement_cannot_raise_accuracy_sample_count(self):
        matrix=np.tile(np.linspace(-.1,.1,55)[:,None],(1,3))
        a=paired_ni(matrix,-.3)
        b=paired_ni(matrix[:,:1],-.3)
        self.assertEqual(a['datasets'],b['datasets'])
        self.assertAlmostEqual(a['p_value'],b['p_value'])

    def test_cap_retains_rare_classes_and_exact_size(self):
        labels=np.array([0]*1000+[1]*4+[2])
        selected=capped_indices(labels,512,np.random.default_rng(4))
        self.assertEqual(len(selected),512)
        self.assertEqual(len(np.unique(selected)),512)
        self.assertEqual(set(labels[selected]),{0,1,2})


if __name__=='__main__':unittest.main()
