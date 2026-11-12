"""Summarize paired measurements without conflating resample 0 with paper means."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(frame, formats=None):
    formats = formats or {}
    rows = []
    for _, row in frame.iterrows():
        rows.append('| ' + ' | '.join(formats.get(c, str)(row[c]) for c in frame.columns) + ' |')
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join('---' for _ in frame.columns) + ' |'] + rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/optimization_comparison/50k_split42')
    parser.add_argument('--report-dir', default='reports/optimization_comparison')
    args = parser.parse_args()
    output = Path(args.output)
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((output / 'protocol.json').read_text())
    choice = json.loads((output / 'choice.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((output / 'evaluation').glob('*.json'))]
    raw = pd.DataFrame(records)
    raw.to_csv(output / 'measurements.csv', index=False)
    successful = raw[raw.status == 'ok']
    metric_cols = ['accuracy', 'alpha', 'features', 'train_feature_bytes', 'max_prediction_feature_bytes',
                   'fit_s', 'predict_s', 'total_s', 'classifier_fit_s', 'classifier_predict_s',
                   'train_transform_s', 'test_transform_s', 'peak_rss_bytes']
    medians = successful.groupby(['dataset', 'variant'])[metric_cols].median().reset_index()
    medians['fit_other_s'] = medians.fit_s - medians.classifier_fit_s - medians.train_transform_s
    medians['predict_other_s'] = medians.predict_s - medians.classifier_predict_s - medians.test_transform_s
    counts = successful.groupby(['dataset', 'variant']).size().rename('repeats_completed').reset_index()
    medians = medians.merge(counts)
    held = medians[medians.dataset.isin(protocol['split']['heldout'])].copy()
    expected = len(protocol['split']['heldout']) * len(protocol['variants']) * protocol['repeats']
    completed = len(successful[successful.dataset.isin(protocol['split']['heldout'])])
    baseline = held[held.variant == 'baseline'].set_index('dataset')
    summaries = []
    paired_rows = []
    for variant in protocol['variants']:
        current = held[held.variant == variant].set_index('dataset')
        common = baseline.index.intersection(current.index)
        b, v = baseline.loc[common], current.loc[common]
        if len(common) == 0:
            continue
        delta = v.accuracy - b.accuracy
        row = dict(variant=variant, datasets=len(common), accuracy=float(v.accuracy.mean()),
                   delta_pp=float(delta.mean() * 100), wins=int((delta > 0).sum()),
                   ties=int((delta == 0).sum()), losses=int((delta < 0).sum()),
                   total_speedup=float(b.total_s.sum() / v.total_s.sum()),
                   median_dataset_speedup=float((b.total_s / v.total_s).median()),
                   classifier_speedup=float(b.classifier_fit_s.sum() / v.classifier_fit_s.sum()),
                   median_rss_reduction_pct=float((1 - v.peak_rss_bytes / b.peak_rss_bytes).median() * 100),
                   baseline_total_s=float(b.total_s.sum()), total_s=float(v.total_s.sum()))
        for column in ('fit_s', 'predict_s', 'classifier_fit_s', 'classifier_predict_s',
                       'train_transform_s', 'test_transform_s', 'fit_other_s', 'predict_other_s'):
            row[column] = float(v[column].sum())
            row[column + '_speedup'] = float(b[column].sum() / v[column].sum()) if v[column].sum() else np.nan
        row['median_peak_rss_gib'] = float((v.peak_rss_bytes / 1024**3).median())
        row['features'] = int(v.features.median())
        row['median_train_feature_mib'] = float((v.train_feature_bytes / 1024**2).median())
        row['median_prediction_feature_mib'] = float((v.max_prediction_feature_bytes / 1024**2).median())
        rng = np.random.default_rng(19)
        means = rng.choice(delta.to_numpy(), size=(10000, len(delta)), replace=True).mean(axis=1)
        row['delta_pp_bootstrap_95'] = (np.quantile(means, [0.025, 0.975]) * 100).tolist()
        summaries.append(row)
        for dataset in common:
            r = dict(dataset=dataset, variant=variant, accuracy=v.loc[dataset, 'accuracy'],
                     baseline_accuracy=b.loc[dataset, 'accuracy'], delta_pp=delta.loc[dataset] * 100,
                     total_speedup=b.loc[dataset, 'total_s'] / v.loc[dataset, 'total_s'])
            r.update(v.loc[dataset].to_dict())
            paired_rows.append(r)
    summary = pd.DataFrame(summaries)
    paired = pd.DataFrame(paired_rows)
    # Published values are contextual, never substituted for the paired baseline.
    budget = protocol['num_features']
    paper_column = {50000: 'MultiRocket', 10000: 'MultiRocket_10k', 100000: 'MultiRocket_100k'}[budget]
    paper_means = pd.read_csv('results/all_resamples_average.csv').set_index('dataset_name')[paper_column]
    paper_folds = pd.read_csv(f'results/resamples/MULTIROCKET_{budget//1000}K_TESTFOLDS.csv').set_index('folds:')['0']
    paired['paper_resample0_accuracy'] = paired.dataset.map(paper_folds)
    paired['paper_30_resample_mean_accuracy'] = paired.dataset.map(paper_means)
    historical = {}
    for dataset in protocol['split']['heldout'] + ['ElectricDevices']:
        path = Path(f'output/multirocket/resample_0/MultiRocket_{budget}/{dataset}/results.csv')
        if path.exists():
            historical[dataset] = pd.read_csv(path).iloc[0].to_dict()
    paired['historical_local_accuracy'] = paired.dataset.map({k: v['test_acc'] for k, v in historical.items()})
    paired['historical_local_classifier_fit_s'] = paired.dataset.map({k: v['train_time'] for k, v in historical.items()})
    for column, old_column in (('classifier_predict_s', 'test_time'),
                               ('train_transform_s', 'apply_kernel_on_train_time'),
                               ('test_transform_s', 'apply_kernel_on_test_time'),
                               ('kernel_generation_s', 'generate_kernel_time'),
                               ('train_difference_s', 'train_transform_time'),
                               ('test_difference_s', 'test_transform_time')):
        paired['historical_local_' + column] = paired.dataset.map(
            {k: v[old_column] for k, v in historical.items()})
    paired.to_csv(report / 'per_dataset.csv', index=False)
    summary.to_csv(report / 'summary.csv', index=False)
    stages = summary[['variant', 'datasets', 'features', 'fit_s', 'predict_s', 'total_s',
                      'train_transform_s', 'test_transform_s', 'classifier_fit_s',
                      'classifier_predict_s', 'fit_other_s', 'predict_other_s',
                      'median_peak_rss_gib', 'median_train_feature_mib',
                      'median_prediction_feature_mib']].copy()
    stages.to_csv(report / 'stage_times_and_memory.csv', index=False)
    comparison_names = baseline.index
    context = pd.DataFrame([
        dict(reference='Published MultiRocket, resample 0', datasets=len(comparison_names),
             accuracy=paper_folds.reindex(comparison_names).mean()),
        dict(reference='Published MultiRocket, 30-resample mean', datasets=len(comparison_names),
             accuracy=paper_means.reindex(comparison_names).mean()),
        dict(reference='Existing local 50k results, resample 0',
             datasets=sum(d in historical for d in comparison_names),
             accuracy=np.mean([historical[d]['test_acc'] for d in comparison_names if d in historical])),
        dict(reference='Fresh seeded baseline, resample 0', datasets=len(baseline), accuracy=baseline.accuracy.mean()),
    ])
    context.to_csv(report / 'reference_metrics.csv', index=False)
    # Check prediction identity, not just equal accuracy.
    parity = []
    for fast, reference in [('ridge_only', 'baseline'), ('memory_only', 'baseline'),
                            ('pooling_ridge', 'pooling_only'), ('combined', 'pooling_only')]:
        for dataset in protocol['split']['heldout'] + ['ElectricDevices']:
            a = output / 'evaluation' / f'{dataset}__{fast}__0'
            b = output / 'evaluation' / f'{dataset}__{reference}__0'
            if not a.with_suffix('.npy').exists() or not b.with_suffix('.npy').exists():
                continue
            pa, pb = np.load(a.with_suffix('.npy')), np.load(b.with_suffix('.npy'))
            ma, mb = json.loads(a.with_suffix('.json').read_text()), json.loads(b.with_suffix('.json').read_text())
            parity.append(dict(dataset=dataset, variant=fast, reference=reference,
                               alpha_equal=ma['alpha'] == mb['alpha'], prediction_agreement=float(np.mean(pa == pb))))
    pd.DataFrame(parity).to_csv(report / 'parity.csv', index=False)
    stress = medians[medians.dataset == 'ElectricDevices'].copy()
    stress.to_csv(report / 'electricdevices.csv', index=False)
    stress_records = raw[raw.dataset == 'ElectricDevices']
    stress_status = pd.DataFrame([
        dict(variant=variant, attempts=int(len(group)), successful=int((group.status == 'ok').sum()),
             rss_limit=int((group.status == 'rss_limit').sum()),
             other_failures=int((~group.status.isin(['ok', 'rss_limit'])).sum()),
             largest_observed_rss_gib=float(group.peak_rss_bytes.max() / 1024**3))
        for variant, group in stress_records.groupby('variant')
    ])
    stress_status.to_csv(report / 'electricdevices_status.csv', index=False)
    incomplete = raw[raw.status != 'ok'].to_dict('records')
    payload = dict(choice=choice, expected_heldout_runs=expected, completed_heldout_runs=completed,
                   summaries=summaries, parity=parity, failures=incomplete)
    (report / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    table = summary[['variant', 'datasets', 'accuracy', 'delta_pp', 'wins', 'ties', 'losses',
                     'classifier_speedup', 'total_speedup', 'median_rss_reduction_pct']].copy()
    candidates = pd.DataFrame([{'operators': k, 'selection_accuracy': v}
                               for k, v in choice['candidate_means'].items()]).sort_values(
                                   'selection_accuracy', ascending=False)
    margin_pp = 100 * (candidates.iloc[0].selection_accuracy - candidates.iloc[1].selection_accuracy)
    combined = summary[summary.variant == 'combined']
    findings = []
    if len(combined):
        c = combined.iloc[0]
        ci = c.delta_pp_bootstrap_95
        findings = [
            '## What improved and what changed', '',
            f"Across the {int(c.datasets)} paired held-out datasets, all three changes together give **{c.classifier_speedup:.2f}x classifier-fit speedup** and **{c.total_speedup:.2f}x end-to-end speedup** (sums of per-dataset median times). Mean accuracy changes by **{c.delta_pp:+.3f} percentage points**; dataset-bootstrap 95% interval **[{ci[0]:+.3f}, {ci[1]:+.3f}] pp**.",
            f"The reduced transform allocates exactly half as many feature columns. Median per-dataset process peak RSS changes by **{-c.median_rss_reduction_pct:+.1f}%**. Reducing feature columns does not imply halving total memory or runtime: unchanged convolution work, Python/library memory, and the sample-space eigendecomposition remain.",
            'The isolated rows above distinguish the effects: pooling changes the representation and can change accuracy; the ridge and memory rows test computational changes on the same representation. Prediction/alpha parity is reported below, so equal aggregate accuracy cannot hide changed predictions.', '']
    text = [
        '# MultiRocket optimization comparison', '',
        f"Selected pair: **{'+'.join(choice['operators'])}**, chosen on all {choice['selection_datasets']} selection datasets.",
        f"Held-out measurements completed: **{completed}/{expected}** ({len(protocol['split']['heldout'])} datasets, {protocol['repeats']} timing repetition{'s' if protocol['repeats'] != 1 else ''}, six variants).",
        'Only complete dataset/variant pairs enter each summary; counts are shown. Timing repetitions reuse the same seed and split, so they are not independent accuracy resamples.', '',
        markdown_table(table, {'accuracy': lambda x: f'{x*100:.3f}%', 'delta_pp': lambda x: f'{x:+.3f}',
                               'classifier_speedup': lambda x: f'{x:.2f}x', 'total_speedup': lambda x: f'{x:.2f}x',
                               'median_rss_reduction_pct': lambda x: f'{0 if abs(x) < 0.05 else x:.1f}%'}), '',
        'Speedups are ratios of summed per-dataset median times against fresh seeded baseline runs on the same machine. RSS reduction is the median per-dataset reduction in process peak RSS; it includes imports and inputs. Peak RSS and feature-array size are different measures.', '',
        'The archived three-repetition run on the same held-out split provides timing context; its fast ridge alpha differed from sklearn on one near tie, although predictions matched. The corrected single-repetition run above resolves that alpha choice. Small timing ratios vary between runs, so treat marginal speed changes cautiously. See `../optimization_comparison_v1/report.md`.', '',
        '## Runtime and memory by stage', '',
        'Times below sum per-dataset medians over the same paired datasets in each row. "Other fit" includes kernel generation, differencing and feature cleaning; "other predict" includes differencing, cleaning and batch overhead. Scaling is included in classifier fit/predict. Seconds are shown throughout.', '',
        markdown_table(stages[['variant', 'fit_s', 'train_transform_s',
                               'classifier_fit_s', 'fit_other_s']].rename(columns={
                                   'fit_s': 'fit total (s)',
                                   'train_transform_s': 'train features (s)',
                                   'classifier_fit_s': 'scale + ridge fit (s)',
                                   'fit_other_s': 'other fit (s)'}),
                       {c: (lambda x: f'{x:.2f}') for c in
                        ('fit total (s)', 'train features (s)', 'scale + ridge fit (s)',
                         'other fit (s)')}), '',
        markdown_table(stages[['variant', 'predict_s', 'test_transform_s',
                               'classifier_predict_s', 'predict_other_s', 'total_s']].rename(columns={
                                   'predict_s': 'predict total (s)',
                                   'test_transform_s': 'test features (s)',
                                   'classifier_predict_s': 'scale + ridge predict (s)',
                                   'predict_other_s': 'other predict (s)',
                                   'total_s': 'fit + predict (s)'}),
                       {c: (lambda x: f'{x:.2f}') for c in
                        ('predict total (s)', 'test features (s)', 'scale + ridge predict (s)',
                         'other predict (s)', 'fit + predict (s)')}), '',
        'Feature memory is the median per-dataset feature-array size. Peak RSS is the median total process peak and includes imports, loaded data, and solver workspace.', '',
        markdown_table(stages[['variant', 'features', 'median_train_feature_mib',
                               'median_prediction_feature_mib', 'median_peak_rss_gib']].rename(columns={
                                   'median_train_feature_mib': 'train features (MiB)',
                                   'median_prediction_feature_mib': 'largest test batch (MiB)',
                                   'median_peak_rss_gib': 'peak RSS (GiB)'}),
                       {c: (lambda x: f'{x:.2f}') for c in
                        ('train features (MiB)', 'largest test batch (MiB)',
                         'peak RSS (GiB)')}), '',
        f"The per-dataset CSV also includes alpha, published and local accuracy, plus the older repo's recorded classifier, transform, kernel-generation, and differencing times. Prediction agreement is in parity.csv. Raw {protocol['repeats']}-repeat records and predictions remain in the experiment output.", '',
        *findings,
        '## Protocol and isolation', '',
        f"Nominal feature budget {budget:,}: all four operators produce {budget//8//84*84*8:,} columns; two produce {budget//8//84*84*4:,}. The raw and first-difference kernels, biases, data split, and per-dataset seed are shared across variants.",
        f"The fixed split has 54 selection and 55 held-out datasets (split seed {protocol['split_seed']}); the earlier partial run's split was retained but its scores were not reused. Six pairs compete by unweighted mean selection accuracy. The choice and selection-file hashes are frozen before evaluation. Selection uses the fast solver's same LOO-MSE objective; evaluation includes sklearn references.",
        f"Resample 0 only, {protocol['threads']} Numba threads, one BLAS thread, warm JIT, prediction batches of {protocol['batch_size']}. Per-job RSS guard: {protocol['max_rss_gib']} GiB. Guarded terminations are reported as resource-limit failures, not OS OOMs. Hardware: {protocol['platform']}; sklearn {protocol['sklearn']}, NumPy {protocol['numpy']}, SciPy {protocol['scipy']}, Numba {protocol['numba']}.",
        'Variant definitions: baseline = four operators + sklearn + original copying/full prediction; pooling_only = selected pair; ridge_only = fast solver; memory_only = in-place cleaning/scaling + chunked prediction; pooling_ridge = pair + fast solver; combined = all three.', '',
        markdown_table(candidates, {'selection_accuracy': lambda x: f'{x*100:.6f}%'}), '',
        f'The winning margin over the next pair is only {margin_pp:.6f} percentage points on the selection half. This is a fragile ranking; the pair is nevertheless kept fixed for the independent held-out comparison. No held-out result is used to revise it.', '',
        '## Paper and repository context', '',
        'The attached paper (arXiv:2102.00457v4, sections 3.5, 4.1 and 4.3.2, Table 2, Figure 10) uses 49,728 features by default, 30 resamples for the 109-dataset accuracy comparison, and a separate 112-problem single-resample timing table (15.77 minutes training at 50k). Its hardware was AMD EPYC 7702; the main experiments used 32 threads and 64 GB RAM. Its pooling design was tuned on 40 development datasets, and all four operators ranked best in that ablation.',
        'The per-dataset CSV includes the published fold-0 accuracy, published 30-resample mean, and the existing local result where available. Those contextual columns have different random kernels and, for the mean, different train/test splits. They cannot establish an improvement by themselves; fresh paired runs are the causal comparison. No cross-machine speedup against the paper is claimed.',
        'In the local repository CSVs, train_time measures scaler/classifier fitting, rather than full transformation plus fitting. The new measurements record both classifier_fit_s and complete fit/predict wall-clock times, including feature cleaning.',
        markdown_table(context, {'accuracy': lambda x: f'{x*100:.3f}%'}),
        'All reference averages above use the same held-out dataset names. Figure 10 is context for operator diversity; it is not a published result for this fixed-kernel, half-feature experiment.',
        'The new fast solver matches the default sklearn leave-one-out squared-error criterion, including the unpenalized intercept. It uses float64 blocked Gram accumulation and fits coefficients only for the selected alpha. For wide matrices, it still requires O(n_train²) memory and O(n_train³) eigendecomposition. Tall matrices fall back to sklearn.',
        'Memory changes remove cleaning/scaling output copies and bound test feature allocation by the batch size. The original class already kept training features local to fit; explicit deletion makes the lifetime clear but is not a new elimination of persistent cached features.', '',
        '## Alpha and prediction parity', '',
        markdown_table(pd.DataFrame(parity).groupby(['variant', 'reference']).agg(
            datasets=('dataset', 'size'), alpha_matches=('alpha_equal', 'sum'),
            minimum_prediction_agreement=('prediction_agreement', 'min')).reset_index(),
            {'minimum_prediction_agreement': lambda x: f'{100*x:.6f}%'}), '',
        'These are observed parity results, not a universal guarantee of bitwise identity on every numerical edge case. Unit checks also cover binary/multiclass targets, rank deficiency, constant columns, constant matrices, float32/float64, and the tall-data sklearn fallback.', '',
        '## ElectricDevices stress case', '',
        'ElectricDevices belongs to the selection half, so it is excluded from held-out accuracy aggregates. An existing successful local 50k run already exists; an historical OOM is not independently established by the available artifacts.',
    ]
    if len(stress):
        st = stress[['variant', 'accuracy', 'features', 'total_s', 'classifier_fit_s', 'peak_rss_bytes',
                     'train_feature_bytes', 'max_prediction_feature_bytes']].copy()
        text += [markdown_table(st, {'accuracy': lambda x: f'{x*100:.3f}%',
                                    'total_s': lambda x: f'{x:.3f}', 'classifier_fit_s': lambda x: f'{x:.3f}',
                                    'peak_rss_bytes': lambda x: f'{x/1024**3:.3f} GiB',
                                    'train_feature_bytes': lambda x: f'{x/1024**2:.1f} MiB',
                                    'max_prediction_feature_bytes': lambda x: f'{x/1024**2:.1f} MiB'}), '']
    if len(stress_status):
        text += ['The table above summarizes successful runs only. Every attempt, including guard stops, is counted here. A guard stop means the 9 GiB process limit was crossed; it does not establish an operating-system out-of-memory crash.', '',
                 markdown_table(stress_status,
                                {'largest_observed_rss_gib': lambda x: f'{x:.2f} GiB'}), '']
    old_stress = historical.get('ElectricDevices')
    stress_refs = pd.DataFrame([
        dict(reference='Paper/repo published resample 0', accuracy=float(paper_folds['ElectricDevices']),
             classifier_fit_s=np.nan),
        dict(reference='Paper/repo published 30-resample mean', accuracy=float(paper_means['ElectricDevices']),
             classifier_fit_s=np.nan),
        dict(reference='Existing local 50k run', accuracy=float(old_stress['test_acc']) if old_stress else np.nan,
             classifier_fit_s=float(old_stress['train_time']) if old_stress else np.nan),
    ])
    stress_refs.to_csv(report / 'electricdevices_references.csv', index=False)
    text += ['Published and earlier local ElectricDevices accuracy is contextual; the old local fit time records scaler/classifier fit only and has no peak-RSS measurement.', '',
             markdown_table(stress_refs, {'accuracy': lambda x: f'{x*100:.3f}%',
                                           'classifier_fit_s': lambda x: '—' if pd.isna(x) else f'{x:.2f}'}), '']
    changes = paired[paired.variant == 'combined'].sort_values('delta_pp')
    if len(changes):
        extreme = pd.concat([changes.head(5), changes.tail(5)]).drop_duplicates('dataset')
        text += ['## Dataset-level gains and losses', '',
                 'Five lowest and five highest paired accuracy changes for the combined variant (see per_dataset.csv for every dataset):', '',
                 markdown_table(extreme[['dataset', 'baseline_accuracy', 'accuracy', 'delta_pp', 'total_speedup']],
                                {'baseline_accuracy': lambda x: f'{x*100:.3f}%',
                                 'accuracy': lambda x: f'{x*100:.3f}%', 'delta_pp': lambda x: f'{x:+.3f}',
                                 'total_speedup': lambda x: f'{x:.2f}x'}), '']
    text += ['## Limits and reproducibility', '',
             'This is a single train/test split per dataset, not the paper\'s full 30-resample replication. Dataset-bootstrap confidence intervals for mean paired accuracy differences are in summary.json; they describe variability across this held-out collection, not across training resamples. Pair selection and timings may vary with seeds, versions, and hardware.',
             'The frozen protocol, dataset/source SHA-256 hashes, selected pair, all raw timing/RSS records, predictions, and worker logs are in `' + str(output) + '`. Per-dataset, stage, parity, ElectricDevices and summary CSV files are beside this report.',
             'Run: `.venv/bin/python -m experiments.compare_optimizations --output ' + str(output) + ' --repeats ' + str(protocol['repeats']) + '` (resumes matching completed jobs); then `.venv/bin/python -m experiments.summarize_optimizations --output ' + str(output) + '`. Use a new output directory after source changes. Run tests with `.venv/bin/python -m unittest discover -s tests -v`.', '',
             'Job failures: ' + (', '.join(f'{row["dataset"]} {row["variant"]} repeat {row["repeat"]}: {row["status"]}' for row in incomplete) if incomplete else 'none.') + ' Full records are in electricdevices_status.csv and the raw experiment output.', '']
    (report / 'report.md').write_text('\n'.join(text))
    print(markdown_table(table))
    print(f'Wrote {report}/report.md')


if __name__ == '__main__':
    main()
