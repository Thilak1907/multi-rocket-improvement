"""Summarize the frozen view-ensemble experiment without retuning its winner."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.accuracy_search import collect, VIEWS
from experiments.compare_optimizations import digest


def plot_confirmation(per, selected, report):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/multirocket-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = per.pivot(index='dataset', columns='variant', values='accuracy') * 100
    delta = (values[selected] - values.baseline).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(values.baseline, values[selected], s=25, alpha=.8)
    floor = max(0, float(values.min().min()) - 3)
    axes[0].plot([floor, 100], [floor, 100], color='gray', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Single-model accuracy (%)', ylabel='Two-view accuracy (%)',
                title='55 datasets, three resamples averaged', xlim=(floor, 101), ylim=(floor, 101))
    axes[1].bar(np.arange(len(delta)), delta, color=np.where(delta >= 0, '#287D8E', '#BD594B'))
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set(xlabel='Datasets sorted by paired change', ylabel='Accuracy change (percentage points)',
                title='Mean gains can hide dataset-specific losses')
    fig.tight_layout()
    fig.savefig(report / 'accuracy_comparison.png', dpi=160)
    fig.savefig(report / 'accuracy_comparison.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/accuracy_search/views_v1')
    parser.add_argument('--report', default='reports/accuracy_search')
    args = parser.parse_args()
    out, report = Path(args.output), Path(args.report)
    report.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((out / 'protocol.json').read_text())
    selection = collect(out, 'selection')
    selection.to_csv(report / 'selection_raw.csv', index=False)
    text = ['# Accuracy addition: two-view ensemble', '',
            'The baseline remains the corrected 30k bundle: all four pooling operators, blocked ridge, buffered extraction, in-place processing, 256-row prediction batches, and folded scaling.', '',
            'Each candidate adds one more complete 30k member and averages its ridge decision scores equally with the baseline. Total feature columns become 59,136 across two members, with separate ridge fits; this is extra work, not a same-cost accuracy improvement.', '',
            'Candidates: another kernel seed; per-series z-normalization (constants become zeros); and pair-averaged, half-rate series (lengths below 20 retain the original series). The added member uses seed +104729 modulo 2^32 in every candidate.', '',
            '## Selection', '',
            'Candidates are compared on stratified 80/20 partitions of the official TRAIN files for the 54 selection datasets, capped at 512 series per partition. Official TEST labels are not used for choosing a candidate. The winner must improve mean validation accuracy by at least 0.1 percentage points and take at most 2.5 times baseline summed runtime. Choice, input hashes, and sources are frozen before confirmation.', '']
    if len(selection) and (selection.status == 'ok').all():
        pivot = selection.pivot(index='dataset', columns='variant', values='accuracy')
        times = selection.groupby('variant').total_s.sum()
        summaries = []
        for view in VIEWS:
            summaries.append(dict(variant=view, datasets=len(pivot), accuracy_pct=100 * pivot[view].mean(),
                                  delta_pp=100 * (pivot[view] - pivot.baseline).mean(), runtime_ratio=times[view] / times.baseline))
        pd.DataFrame(summaries).to_csv(report / 'selection_summary.csv', index=False)
        text += ['| Addition | Validation accuracy | Change | Runtime / baseline |', '| --- | ---: | ---: | ---: |']
        for r in summaries:
            text += [f"| {r['variant']} | {r['accuracy_pct']:.3f}% | {r['delta_pp']:+.3f} pp | {r['runtime_ratio']:.2f}× |"]
    choice_file = out / 'choice.json'
    choice = json.loads(choice_file.read_text()) if choice_file.exists() else {'selected': None}
    decision = dict(adopt=False, status='pending', selected=choice['selected'],
                    protocol_sha256=digest(out / 'protocol.json'),
                    analysis_source_sha256=digest(Path(__file__)),
                    choice_sha256=digest(choice_file) if choice_file.exists() else None)
    if choice_file.exists() and not choice['selected']:
        decision.update(status='retain_previous', reason='No candidate met the selection gate')
        text += ['', '**No candidate met the selection gate. Retain the existing bundle.**']
    elif choice['selected']:
        text += ['', f"Frozen candidate: **{choice['selected']}**.", '', '## Paired confirmation', '']
        frame = collect(out, 'confirmation')
        frame.to_csv(report / 'confirmation_raw.csv', index=False)
        complete = len(frame) == 55 * 3 * 2 and (frame.status == 'ok').all()
        if complete:
            pivot = frame.pivot(index=['dataset', 'resample'], columns='variant', values='accuracy')
            delta = (pivot[choice['selected']] - pivot.baseline).unstack('resample') * 100
            assert delta.shape == (55, 3) and not delta.isna().any().any()
            delta.to_csv(report / 'paired_differences.csv')
            differences = delta.mean(axis=1).to_numpy()
            rng = np.random.default_rng(421)
            draws = differences[rng.integers(0, 55, size=(20000, 55))].mean(axis=1)
            low, high = np.percentile(draws, [2.5, 97.5])
            times = frame.groupby('variant').total_s.sum()
            ratio = float(times[choice['selected']] / times.baseline)
            mean = float(differences.mean())
            adopt = mean >= .1 and low > 0 and ratio <= 2.5
            decision.update(adopt=bool(adopt), status='recommend_optional' if adopt else 'retain_previous',
                            mean_delta_pp=mean, ci_low_pp=float(low), ci_high_pp=float(high),
                            runtime_ratio=ratio, datasets=55, resamples=3)
            means = frame.groupby('variant')[['accuracy', 'balanced_accuracy', 'macro_f1', 'fit_s', 'predict_s', 'model_bytes', 'process_highwater_bytes']].mean()
            means.to_csv(report / 'confirmation_summary.csv')
            per = frame.groupby(['dataset', 'variant'])[['accuracy', 'balanced_accuracy', 'macro_f1', 'total_s', 'model_bytes', 'process_highwater_bytes']].mean().reset_index()
            per.to_csv(report / 'per_dataset.csv', index=False)
            wins, ties, losses = int((differences > 1e-10).sum()), int((np.abs(differences) <= 1e-10).sum()), int((differences < -1e-10).sum())
            text += [f'On 55 datasets across archive resamples 1, 2, and 3: **{mean:+.3f} percentage points**, paired dataset-bootstrap 95% CI **[{low:+.3f}, {high:+.3f}]**. Datasets, not 165 dataset/resample rows, are the statistical units. Bootstrap: 20,000 draws, seed 421.', '',
                     f'Summed fit-plus-predict runtime: **{ratio:.2f}× baseline**. Per-dataset wins/ties/losses after averaging three resamples: **{wins}/{ties}/{losses}** (tie tolerance 1e-10 percentage points).', '',
                     '| Model | Accuracy | Balanced accuracy | Macro F1 | Mean fit s | Mean predict s | Mean model MiB |',
                     '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for name, r in means.iterrows():
                text += [f'| {name} | {100*r.accuracy:.3f}% | {100*r.balanced_accuracy:.3f}% | {100*r.macro_f1:.3f}% | {r.fit_s:.3f} | {r.predict_s:.3f} | {r.model_bytes/1024**2:.2f} |']
            plot_confirmation(per, choice['selected'], report)
            text += ['', '![Paired accuracy comparison](accuracy_comparison.png)', '', '**Decision: ' + ('offer this as an optional accuracy-focused configuration; retain the faster single-model preset as the efficiency default.' if adopt else 'retain the existing bundle; the candidate did not pass the predeclared gain/runtime gate.') + '**']
        else:
            text += [f'Confirmation incomplete: {len(frame)}/330 result rows. No adoption decision yet.']
    else:
        text += ['', 'Selection is still running; no winner has been chosen.']
    text += ['', '## Costs and limits', '',
             'Timing has one run per split; small speed differences are descriptive. Baseline features/scores are reused across candidates, and candidate order is rotated by dataset. Imports, loading, and warmup are excluded. Input-view processing is included in candidate fit/predict times. Models fit sequentially and do not retain training features.', '',
             'Process RSS is a shared high-water upper bound, including imports/warmup and previous candidate fits; it is not an isolated per-candidate memory measurement. Every job has a 9 GiB polling guard. Reported model bytes sum the serialized members, excluding small wrapper metadata.', '',
             'The confirmation dataset names were inspected during previous development. These are follow-up paired measurements, not untouched external validation, and resamples reuse examples. No claim of a universal accuracy gain, energy efficiency, or GPU performance is made.', '',
             'The current production preset is not changed by this experiment. The optional implementation is `multirocket/accuracy_ensemble.py`. Reproduce with `python -m experiments.accuracy_search`, then `python -m experiments.summarize_accuracy`. Raw jobs, frozen choice, source snapshots, and hashes are under `output/accuracy_search/views_v1`.', '',
             'Background: [MultiRocket paper](https://arxiv.org/abs/2102.00457).']
    (report / 'report.md').write_text('\n'.join(text) + '\n')
    (report / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
