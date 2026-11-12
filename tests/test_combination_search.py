import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.search_combinations import CONFIGS, data_for, selection_summary
from experiments.summarize_combinations import summarize


class CombinationProtocolTests(unittest.TestCase):
    def test_validation_never_uses_official_test_and_keeps_singletons(self):
        X = np.arange(22).reshape(11,2).astype(float)
        y = np.array([0]*5 + [1]*5 + [2])
        Xt = np.full((6,2), 999.)
        with patch('experiments.search_combinations.load_data',return_value=(X,y,Xt,np.zeros(6))):
            train, labels, val, targets, _ = data_for('Example','selection',{'seed':3})
        self.assertEqual(len(train)+len(val),len(X))
        self.assertEqual(set(train[:,0]) & set(val[:,0]),set())
        self.assertNotIn(999,train)
        self.assertNotIn(999,val)
        self.assertIn(2,labels)
        self.assertNotIn(2,targets)

    def test_balanced_choice_rejects_fast_inaccurate_model_and_freezes(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            (output/'selection').mkdir()
            protocol={'split':{'selection':['A','B']},'accuracy_tolerance_pp':.5,
                      'balanced_rule':'test'}
            for dataset in ('A','B'):
                for variant in CONFIGS:
                    row=dict(dataset=dataset,variant=variant,status='ok',accuracy=.9,
                             total_s=1.,peak_rss_bytes=100)
                    if variant=='fused10':
                        row.update(accuracy=.8,total_s=.01,peak_rss_bytes=10)
                    if variant=='fused20':
                        row.update(accuracy=.897,total_s=.5,peak_rss_bytes=50)
                    if variant=='fused30':
                        row.update(accuracy=.901,total_s=.7,peak_rss_bytes=70)
                    (output/'selection'/f'{dataset}__{variant}__0.json').write_text(json.dumps(row))
            choice=selection_summary(output,protocol)
            self.assertEqual(choice['balanced'],'fused20')
            self.assertNotIn('fused10',choice['finalists'])
            self.assertEqual(choice,selection_summary(output,protocol))
            from experiments.repeat_confirmation import freeze_conservative
            (output/'protocol.json').write_text(json.dumps(protocol))
            self.assertEqual(freeze_conservative(output)['conservative'],'fused30')
            target=output/'selection'/'A__baseline50__0.json'
            row=json.loads(target.read_text())
            row['accuracy']=.1
            target.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                selection_summary(output,protocol)

    def test_summary_pairs_do_not_drop_changed_predictions_or_metrics(self):
        import pandas as pd
        base=dict(dataset='A',repeat=0,status='ok',accuracy=.9,balanced_accuracy=.8,
                  macro_f1=.85,fit_s=1.,predict_s=1.,total_s=2.,classifier_fit_s=.5,
                  classifier_predict_s=.1,train_transform_s=.5,test_transform_s=.9,
                  peak_rss_bytes=100,model_bytes=10,features=8,train_feature_bytes=32,
                  max_prediction_feature_bytes=32,n_test=10)
        raw=pd.DataFrame([{**base,'variant':'baseline50'},
                          {**base,'variant':'fused20','accuracy':.8,'total_s':1.,'peak_rss_bytes':50}])
        summary,pairs,_=summarize(raw)
        candidate=summary[summary.variant=='fused20'].iloc[0]
        self.assertAlmostEqual(candidate.delta_pp,-10.)
        self.assertEqual(candidate.speedup,2.)
        self.assertEqual(len(pairs),2)


if __name__=='__main__':
    unittest.main()
