"""Report fixed feature-map selection and paired confirmation, without retuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.feature_power_search import collect, VIEWS
from experiments.compare_optimizations import digest
from experiments.summarize_accuracy import plot_confirmation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/feature_power_search/v1')
    parser.add_argument('--report',default='reports/feature_power_search')
    args=parser.parse_args();out=Path(args.output);report=Path(args.report)
    report.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((out/'protocol.json').read_text())
    keys=['split','seed','threads','blas_threads','views','selection','rule','confirmation','adoption','memory','timing','sources','versions']
    (report/'experiment_plan.json').write_text(json.dumps({k:protocol[k] for k in keys},indent=2)+'\n')
    frame=collect(out,'selection');frame.to_csv(report/'selection_raw.csv',index=False)
    lines=['# Fixed nonlinear feature maps for the retained 30k bundle','',
           'The production default remains unchanged by this experiment. Every model keeps the corrected 30k bundle, all four pooling operators, blocked ridge, buffered extraction, in-place processing, 256-row prediction batches, and folded scaling. Each candidate has 29,568 columns and one ridge head, without an ensemble.','',
           'The sole change is a fixed pointwise feature map before StandardScaler: signed square root `sign(z)*sqrt(abs(z))`, or signed logarithm `sign(z)*log1p(abs(z))`. Maps are applied in place, with the same kernels and seeds as baseline. There are no fitted map parameters or test-distribution statistics. These maps are a hypothesis, not sklearn\'s fitted Yeo-Johnson/Box-Cox PowerTransformer.','',
           'Background: nonlinear feature transforms can change skewness and scale effects ([sklearn preprocessing documentation](https://scikit-learn.org/stable/modules/preprocessing.html#power-transforms)); this does not guarantee improved classification.','',
           '## Frozen selection','',
           '54 selection dataset names; stratified TRAIN-only 80/20 partitions, capped at 512 cases in each partition. Choose the largest mean validation gain, provided it is at least +0.1 percentage points and summed fit-plus-predict runtime is at most 1.20 times baseline. Candidates are not selected using confirmation labels.','']
    if len(frame) and (frame.status=='ok').all():
        pivot=frame.pivot(index='dataset',columns='variant',values='accuracy')
        times=frame.groupby('variant').total_s.sum()
        rows=[dict(variant=v,datasets=len(pivot),accuracy_pct=100*pivot[v].mean(),delta_pp=100*(pivot[v]-pivot.baseline).mean(),runtime_ratio=times[v]/times.baseline) for v in VIEWS]
        pd.DataFrame(rows).to_csv(report/'selection_summary.csv',index=False)
        lines+=['| Map | Validation accuracy | Change | Runtime / baseline |','| --- | ---: | ---: | ---: |']
        for r in rows:lines+=[f"| {r['variant']} | {r['accuracy_pct']:.3f}% | {r['delta_pp']:+.3f} pp | {r['runtime_ratio']:.3f}× |"]
    choice_file=out/'choice.json';choice=json.loads(choice_file.read_text()) if choice_file.exists() else dict(selected=None)
    decision=dict(adopt=False,status='pending',selected=choice['selected'],protocol_sha256=digest(out/'protocol.json'),analysis_source_sha256=digest(Path(__file__)),choice_sha256=digest(choice_file) if choice_file.exists() else None)
    if choice_file.exists() and not choice['selected']:
        decision.update(status='retain_previous',reason='Neither map passed the TRAIN-only selection gate')
        lines+=['','**Neither map passed selection. Retain the current recommendation.**']
    elif choice['selected']:
        name=choice['selected'];lines+=['',f'Frozen winner: **{name}**.','','## Paired confirmation','']
        raw=collect(out,'confirmation');raw.to_csv(report/'confirmation_raw.csv',index=False)
        if len(raw)==330 and (raw.status=='ok').all():
            pivot=raw.pivot(index=['dataset','resample'],columns='variant',values='accuracy')
            delta=((pivot[name]-pivot.baseline)*100).unstack('resample')
            assert delta.shape==(55,3) and not delta.isna().any().any()
            delta.to_csv(report/'paired_differences.csv')
            d=delta.mean(axis=1).to_numpy();rng=np.random.default_rng(421)
            lower,upper=np.percentile(d[rng.integers(0,55,size=(20000,55))].mean(axis=1),[2.5,97.5])
            times=raw.groupby('variant').total_s.sum();ratio=float(times[name]/times.baseline)
            gain=float(d.mean());adopt=bool(gain>=.1 and lower>0 and ratio<=1.2)
            decision.update(adopt=adopt,status='recommend_optional' if adopt else 'retain_previous',mean_delta_pp=gain,ci_low_pp=float(lower),ci_high_pp=float(upper),runtime_ratio=ratio,datasets=55,resamples=3)
            mean=raw.groupby('variant')[['accuracy','balanced_accuracy','macro_f1','fit_s','predict_s','model_bytes','process_highwater_bytes']].mean()
            mean.to_csv(report/'confirmation_summary.csv')
            per=raw.groupby(['dataset','variant'])[['accuracy','balanced_accuracy','macro_f1','total_s','model_bytes','process_highwater_bytes']].mean().reset_index()
            per.to_csv(report/'per_dataset.csv',index=False)
            lines += [f'55 datasets, archive resamples 1–3. Mean paired change **{gain:+.3f} percentage points**, dataset-bootstrap 95% CI **[{lower:+.3f}, {upper:+.3f}]**. Bootstrap: 20,000 draws, seed 421; datasets are the units, not 165 independent rows.','',f'Runtime: **{ratio:.3f}× baseline**. Dataset-mean wins/ties/losses: **{int((d>1e-10).sum())}/{int((abs(d)<=1e-10).sum())}/{int((d< -1e-10).sum())}** (tie tolerance 1e-10 pp).','',
                      '| Model | Accuracy | Balanced accuracy | Macro F1 | Mean fit s | Mean predict s | Mean model MiB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for v,r in mean.iterrows():lines+=[f'| {v} | {100*r.accuracy:.3f}% | {100*r.balanced_accuracy:.3f}% | {100*r.macro_f1:.3f}% | {r.fit_s:.3f} | {r.predict_s:.3f} | {r.model_bytes/1024**2:.3f} |']
            plot_confirmation(per,name,report)
            lines+=['','![Paired accuracy comparison](accuracy_comparison.png)','', '**Decision: '+('passes the frozen accuracy/runtime gate; available as an optional configuration pending independent validation.' if adopt else 'does not pass the frozen accuracy/runtime gate; retain the current recommendation.')+'**']
        else:lines += [f'Confirmation pending: {len(raw)}/330 rows.']
    else:lines+=['','Selection remains incomplete.']
    lines+=['','## Limits and reproduction','',
            'One timing measurement per split; small runtime differences are descriptive. Feature-map processing is included in fit/predict time. Baseline is shared across comparisons and evaluated first; candidate order is rotated. Process high-water RSS includes imports/warmup and earlier fits, so it is an upper bound rather than isolated candidate memory. Every dataset worker has a 9 GiB polling guard.','',
            'Confirmation names were inspected during earlier development; these are follow-up paired measurements, not untouched external validation. Resamples reuse examples. This is a finite benchmark comparison, not a universal accuracy guarantee. No hardware, energy, multivariate, or GPU improvement is established.','',
            'Run `python -m experiments.feature_power_search`, then `python -m experiments.summarize_feature_power`. Required data, archive indices, and previous corrected protocol must be available locally. Raw jobs and frozen sources are under `output/feature_power_search/v1`. Optional model: `multirocket/feature_power.py`. The production preset is unchanged.']
    (report/'report.md').write_text('\n'.join(lines)+'\n')
    (report/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
