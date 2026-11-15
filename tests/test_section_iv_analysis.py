"""Regression checks for the standalone Section IV analysis."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('section_iv', ROOT / 'reproduction/analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def run(stage='B', config='paper50', dataset='Coffee', resample=0, seed=0):
    return dict(stage=stage, config=config, dataset=dataset, resample=resample,
                seed=seed, acc=1., fit_s=.1, predict_s=.2)


class SectionIVAnalysisTests(unittest.TestCase):
    def test_decimal_and_binary_memory_units_and_stage_filter(self):
        rss = pd.DataFrame([
            dict(stage='A', dataset='StarLightCurves', config='paper50', peak_rss_mib=1.),
            dict(stage='C', dataset='StarLightCurves', config='paper50', peak_rss_mib=6088.71875),
        ])
        gb, gib = analysis.peak_memory(rss, 'StarLightCurves', 'paper50')
        self.assertEqual(f'{gb:.1f}', '6.4')
        self.assertEqual(f'{gib:.2f}', '5.95')
        self.assertTrue(all(np.isnan(v) for v in analysis.peak_memory(pd.DataFrame(), 'Crop', 'paper50')))
        with self.assertRaisesRegex(ValueError, 'Multiple Stage C'):
            analysis.peak_memory(pd.concat([rss, rss]), 'StarLightCurves', 'paper50')

    def test_duplicate_successes_are_rejected(self):
        data = pd.DataFrame([run(stage='A'), run(stage='A')])
        with self.assertRaisesRegex(ValueError, 'Duplicate successful runs'):
            analysis.validate_runs(data)

    def test_complete_subset_is_valid_but_unpaired_folds_are_not(self):
        records = [run(config=c, resample=r) for c in ['paper50', 'rec30'] for r in range(30)]
        analysis.validate_runs(pd.DataFrame(records))
        with self.assertRaisesRegex(ValueError, 'Incomplete Stage B comparison'):
            analysis.validate_runs(pd.DataFrame(records[:-1]))
        with self.assertRaisesRegex(ValueError, 'require paper50'):
            analysis.validate_runs(pd.DataFrame(records[30:]))
        with self.assertRaisesRegex(ValueError, 'expected resamples 0–29'):
            analysis.validate_runs(pd.DataFrame(records[:29]))

    def test_small_pvalues_are_not_rendered_as_zero(self):
        result = analysis.table(pd.DataFrame({'wilcoxon_p': [0.00000123, np.nan]}), 3)
        self.assertIn('1.23e-06', result)
        self.assertNotIn('0.000', result)
