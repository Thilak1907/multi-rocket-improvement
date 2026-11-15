#!/usr/bin/env python3
"""Build the paper/project comparison from retained CSVs and author results."""
import argparse
import json
from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'output/paper_comparison')
    parser.add_argument('--project-results', type=Path,
                        default=ROOT / 'reports/review_revision/accuracy_by_resample.csv')
    args = parser.parse_args()
    with zipfile.ZipFile(ROOT / 'reproducibility/records.zip') as archive:
        inputs = json.loads(archive.read('reproducibility/inputs.json'))
        with archive.open('results/all_resamples_average.csv') as stream:
            published = pd.read_csv(stream).set_index('dataset_name')['MultiRocket']
        with archive.open('results/resamples/MULTIROCKET_50K_TESTFOLDS.csv') as stream:
            folds = pd.read_csv(stream).set_index('folds:')
    heldout = inputs['split']['heldout']
    full = heldout + inputs['split']['selection']
    scores = pd.read_csv(args.project_results)
    rows = [
        dict(source='Authors released 50k results', datasets=109, resamples='30', accuracy_pct=100*published.loc[full].mean(), features=49728),
        dict(source='Authors released 50k results, project subset', datasets=55, resamples='30', accuracy_pct=100*published.loc[heldout].mean(), features=49728),
        dict(source='Authors released 50k results, project subset', datasets=55, resamples='1 (index 1)', accuracy_pct=100*folds.loc[heldout,'1'].mean(), features=49728),
    ]
    for name, label in [('baseline50','Project corrected 50k'),('fast30','Project corrected 30k')]:
        part=scores[(scores.variant==name)&scores['resample'].isin([1,2,3])]
        assert len(part)==55*3
        assert set(part.dataset) == set(heldout)
        assert not part.duplicated(['dataset', 'resample']).any()
        rows.append(dict(source=label,datasets=55,resamples='3 (indices 1–3)',accuracy_pct=100*part.accuracy.mean(),features=49728 if name=='baseline50' else 29568))
    out=args.report_dir;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    lines=['# Paper and project comparison','',
           'Published accuracy is calculated from the authors\' released CSVs, not transcribed as a new table from the article. The 55-name subset is the project confirmation set. Kernel seeds, feature correctness mode, and number of resamples differ; these rows are context, not a paired superiority test.','',
           '| Source | Datasets | Resamples | Features | Mean accuracy |','| --- | ---: | --- | ---: | ---: |']
    for r in rows:lines.append(f"| {r['source']} | {r['datasets']} | {r['resamples']} | {r['features']:,} | {r['accuracy_pct']:.3f}% |")
    lines+=['','The paper also reports 15.77 minutes of 50k training over 112 problems in its single-thread timing table (Table 2). Our primary timing aggregate covers 55 datasets with four Numba threads and one BLAS thread on different hardware, so no speed-up against that paper timing is calculated. The project\'s 1.66× figure uses its own paired corrected 50k baseline.','',
            'Sources: [paper](https://doi.org/10.1007/s10618-022-00844-1), [authors\' released results](https://github.com/ChangWeiTan/MultiRocket/tree/3ccaa4f8ed11769b904af885e17018c39c89a2ac/results).']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
