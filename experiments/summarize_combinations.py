"""Report a frozen combination search, its independent resample, and stress jobs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.summarize_optimizations import markdown_table
from experiments.compare_optimizations import digest


def records(folder):
    return pd.DataFrame([json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))])


def summarize(raw, baseline='baseline50'):
    ok = raw[raw.status == 'ok']
    metrics = [c for c in ok.select_dtypes(include=np.number).columns if c != 'repeat']
    medians = ok.groupby(['dataset','variant'])[metrics].median().reset_index()
    b = medians[medians.variant == baseline].set_index('dataset')
    rows, pairs = [], []
    for variant in medians.variant.unique():
        v = medians[medians.variant == variant].set_index('dataset')
        common = b.index.intersection(v.index)
        bv, vv = b.loc[common], v.loc[common]
        if not len(common):
            continue
        delta = (vv.accuracy-bv.accuracy)*100
        rng=np.random.default_rng(20261111)
        boot=rng.choice(delta.to_numpy(),size=(10000,len(delta)),replace=True).mean(axis=1)
        low,high=np.quantile(boot,[.025,.975])
        row=dict(variant=variant,datasets=len(common),accuracy=float(vv.accuracy.mean()),
                 balanced_accuracy=float(vv.balanced_accuracy.mean()),macro_f1=float(vv.macro_f1.mean()),
                 delta_pp=float(delta.mean()),delta_ci_low_pp=float(low),delta_ci_high_pp=float(high),
                 worst_delta_pp=float(delta.min()),best_delta_pp=float(delta.max()),
                 wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),
                 fit_s=float(vv.fit_s.sum()),predict_s=float(vv.predict_s.sum()),total_s=float(vv.total_s.sum()),
                 speedup=float(bv.total_s.sum()/vv.total_s.sum()),
                 classifier_fit_s=float(vv.classifier_fit_s.sum()),
                 classifier_predict_s=float(vv.classifier_predict_s.sum()),
                 train_transform_s=float(vv.train_transform_s.sum()),
                 test_transform_s=float(vv.test_transform_s.sum()),
                 median_rss_reduction_pct=float((1-vv.peak_rss_bytes/bv.peak_rss_bytes).median()*100),
                 median_rss_gib=float(vv.peak_rss_bytes.median()/1024**3),
                 max_rss_gib=float(ok[(ok.variant==variant)&ok.dataset.isin(common)].peak_rss_bytes.max()/1024**3),
                 median_model_mib=float(vv.model_bytes.median()/1024**2),
                 median_features=float(vv.features.median()),
                 median_train_feature_mib=float(vv.train_feature_bytes.median()/1024**2),
                 median_test_batch_mib=float(vv.max_prediction_feature_bytes.median()/1024**2),
                 test_cases_per_s=float(vv.n_test.sum()/vv.predict_s.sum()))
        rows.append(row)
        for d in common:
            pairs.append({**vv.loc[d].to_dict(), 'dataset':d, 'variant':variant,
                          'baseline_accuracy':float(bv.loc[d,'accuracy']),
                          'delta_pp':float(delta.loc[d]),
                          'speedup':float(bv.loc[d,'total_s']/vv.loc[d,'total_s']),
                          'rss_reduction_pct':float((1-vv.loc[d,'peak_rss_bytes']/bv.loc[d,'peak_rss_bytes'])*100)})
    summary=pd.DataFrame(rows)
    if len(summary):
        summary['pareto']=True
        for i,r in summary.iterrows():
            # Strict domination on accuracy, runtime and median process RSS.
            others=summary[(summary.accuracy>=r.accuracy)&(summary.total_s<=r.total_s)&
                           (summary.median_rss_gib<=r.median_rss_gib)]
            if ((others.accuracy>r.accuracy)|(others.total_s<r.total_s)|
                (others.median_rss_gib<r.median_rss_gib)).any():
                summary.loc[i,'pareto']=False
    return summary,pd.DataFrame(pairs),medians


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/combination_search/20261111')
    parser.add_argument('--report-dir',default='reports/combination_search')
    parser.add_argument('--refinement-output')
    args=parser.parse_args()
    output,report=Path(args.output),Path(args.report_dir)
    report.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((output/'protocol.json').read_text())
    choice=json.loads((output/'choice.json').read_text())
    extra_file=output/'extra_choice.json'
    extra=None
    if extra_file.exists():
        extra=json.loads(extra_file.read_text())
        if extra['selection_summary_sha256'] != digest(output/'selection_summary.csv'):
            raise ValueError('Conservative choice selection inputs changed')
        if extra['original_choice_sha256'] != digest(output/'choice.json'):
            raise ValueError('Original frozen choice changed')
        choice['conservative']=extra['conservative']
        choice['finalists']=list(dict.fromkeys(choice['finalists']+[extra['conservative']]))
    select=pd.read_csv(output/'selection_summary.csv')
    refinement=Path(args.refinement_output) if args.refinement_output else None
    ref_protocol=None
    if refinement:
        ref_protocol=json.loads((refinement/'protocol.json').read_text())
        ref_choice=json.loads((refinement/'choice.json').read_text())
        if ref_choice['alpha_matches']!=54 or ref_choice['prediction_matches']!=54:
            raise ValueError('Refinement did not pass the selection parity gate')
        for key in ('sources','split','seed','data_hashes','indices_hashes','repeats'):
            if ref_protocol[key]!=protocol[key]:
                raise ValueError('Refinement protocol differs on '+key)
        choice['refined_conservative']='fast30'
        choice['finalists']=list(dict.fromkeys(choice['finalists']+['fast30']))
        protocol['configs']['fast30']=ref_protocol['configs']['fast30']
        base=records(output/'selection')
        base=base[(base.variant=='baseline50')&(base.status=='ok')].set_index('dataset')
        ref=records(refinement/'selection')
        success=ref[ref.status=='ok'].set_index('dataset')
        names=base.index.intersection(success.index)
        b,v=base.loc[names],success.loc[names]
        tr=float(v.total_s.sum()/b.total_s.sum())
        rr=float((v.peak_rss_bytes/b.peak_rss_bytes).median())
        select=pd.concat([select,pd.DataFrame([dict(variant='fast30',datasets=len(names),
                         accuracy=float(v.accuracy.mean()),delta_pp=float((v.accuracy-b.accuracy).mean()*100),
                         time_ratio=tr,rss_ratio=rr,cost_score=float(np.sqrt(tr*rr)),successful=len(success))])],ignore_index=True)
    select.to_csv(report/'selection_summary.csv',index=False)
    atomic_choice={k:v for k,v in choice.items() if k!='selection_files'}
    atomic_choice['configurations']={v:protocol['configs'][v] for v in choice['finalists']}
    raw=records(output/'confirmation')
    if refinement:
        raw=pd.concat([raw,records(refinement/'confirmation')],ignore_index=True)
    if not len(raw):
        raise ValueError('No confirmation jobs')
    raw.to_csv(report/'raw_confirmation.csv',index=False)
    stability=raw[raw.status=='ok'].groupby(['dataset','variant']).agg(
        completed_repeats=('repeat','size'),distinct_predictions=('prediction_hash','nunique'),
        distinct_alphas=('alpha','nunique')).reset_index()
    stability.to_csv(report/'repeat_stability.csv',index=False)
    by_repeat=raw[raw.status=='ok'].groupby(['variant','repeat']).agg(
        datasets=('dataset','size'),fit_s=('fit_s','sum'),predict_s=('predict_s','sum'),
        total_s=('total_s','sum')).reset_index()
    by_repeat.to_csv(report/'runtime_by_repeat.csv',index=False)
    summary,paired,medians=summarize(raw)
    paper=pd.read_csv('results/resamples/MULTIROCKET_50K_TESTFOLDS.csv').set_index('folds:')['1']
    means=pd.read_csv('results/all_resamples_average.csv').set_index('dataset_name')['MultiRocket']
    paired['published_50k_resample1_accuracy']=paired.dataset.map(paper)
    paired['published_50k_30resample_mean_accuracy']=paired.dataset.map(means)
    paired.to_csv(report/'per_dataset.csv',index=False)
    summary.to_csv(report/'confirmation_summary.csv',index=False)
    medians.to_csv(report/'stage_times_memory.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/multirocket-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for i,(_,r) in enumerate(summary.iterrows()):
        color=plt.get_cmap('tab10')(i%10)
        axes[0].errorbar(r.speedup,r.delta_pp,
                         yerr=[[max(0,r.delta_pp-r.delta_ci_low_pp)],[max(0,r.delta_ci_high_pp-r.delta_pp)]],
                         fmt='o',color=color,capsize=3,label=r.variant)
        axes[1].scatter(r.speedup,r.median_rss_reduction_pct,color=color,s=55)
    for ax in axes:
        ax.axvline(1,color='.65',ls='--',lw=1)
        ax.axhline(0,color='.65',ls='--',lw=1)
        ax.set_xlabel('Full runtime speedup (higher is better)')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Accuracy change (percentage points)')
    axes[0].set_title('Accuracy and runtime; dataset-bootstrap 95% intervals')
    axes[1].set_ylabel('Median paired peak RSS reduction (%)')
    axes[1].set_title('Process memory and runtime')
    axes[0].legend(fontsize=8,loc='best')
    fig.savefig(report/'tradeoffs.png',dpi=180)
    fig.savefig(report/'tradeoffs.svg')
    plt.close(fig)
    failures=[]
    for phase in ('selection','confirmation','stress'):
        frame=records(output/phase)
        if len(frame):
            frame=frame[frame.status!='ok'].copy()
            frame['phase']=phase
            failures.extend(frame.to_dict('records'))
        if refinement:
            frame=records(refinement/phase)
            if len(frame):
                frame=frame[frame.status!='ok'].copy()
                frame['phase']=phase
                frame['source']='fast30 refinement'
                failures.extend(frame.to_dict('records'))
    pd.DataFrame(failures).to_csv(report/'failures.csv',index=False)
    parity=[]
    for variant in ('buffered50','fused50','fast50'):
        for phase in ('selection','confirmation'):
            frame=records(output/phase)
            if not len(frame): continue
            b=frame[(frame.variant=='baseline50')&(frame.status=='ok')].set_index(['dataset','repeat'])
            v=frame[(frame.variant==variant)&(frame.status=='ok')].set_index(['dataset','repeat'])
            for index in b.index.intersection(v.index):
                folder=output/phase
                d,r=index
                pred=np.load(folder/f'{d}__{variant}__{r}.npy')
                ref=np.load(folder/f'{d}__baseline50__{r}.npy')
                parity.append(dict(phase=phase,dataset=d,repeat=r,variant=variant,
                                   reference='baseline50',
                                   alpha_equal=bool(b.loc[index,'alpha']==v.loc[index,'alpha']),
                                   prediction_agreement=float(np.mean(pred==ref))))
    if refinement:
        for phase in ('selection','confirmation'):
            b=records(output/phase)
            b=b[(b.variant=='fused30')&(b.status=='ok')].set_index(['dataset','repeat'])
            v=records(refinement/phase)
            v=v[v.status=='ok'].set_index(['dataset','repeat'])
            for d,r in b.index.intersection(v.index):
                pred=np.load(refinement/phase/f'{d}__fast30__{r}.npy')
                ref=np.load(output/phase/f'{d}__fused30__{r}.npy')
                parity.append(dict(phase=phase,dataset=d,repeat=r,variant='fast30',reference='fused30',
                                   alpha_equal=bool(b.loc[(d,r),'alpha']==v.loc[(d,r),'alpha']),
                                   prediction_agreement=float(np.mean(pred==ref))))
    parity=pd.DataFrame(parity)
    parity.to_csv(report/'parity.csv',index=False)
    fmt=dict(accuracy=lambda x:f'{100*x:.3f}%',balanced_accuracy=lambda x:f'{100*x:.3f}%',
             macro_f1=lambda x:f'{100*x:.3f}%',delta_pp=lambda x:f'{x:+.3f}',
             time_ratio=lambda x:f'{x:.3f}',rss_ratio=lambda x:f'{x:.3f}',
             cost_score=lambda x:f'{x:.3f}',total_s=lambda x:f'{x:.2f}',
             speedup=lambda x:f'{x:.2f}x',median_rss_reduction_pct=lambda x:f'{x:.1f}%',
             median_rss_gib=lambda x:f'{x:.2f}',max_rss_gib=lambda x:f'{x:.2f}',
             median_model_mib=lambda x:f'{x:.2f}',median_train_feature_mib=lambda x:f'{x:.2f}',
             median_test_batch_mib=lambda x:f'{x:.2f}',median_features=lambda x:f'{x:.0f}')
    findings=[]
    for label,variant in [('Balanced',choice['balanced']),('Conservative',choice.get('conservative')),
                          ('Conservative with fast ridge',choice.get('refined_conservative')),
                          ('Accuracy',choice['accuracy_winner'])]:
        current=summary[summary.variant==variant]
        if not len(current): continue
        r=current.iloc[0]
        findings.append(f"- **{label}: {variant}** — {r.accuracy*100:.3f}% accuracy ({r.delta_pp:+.3f} pp), {r.speedup:.2f}x full-runtime speedup, {r.median_rss_reduction_pct:.1f}% median paired peak-RSS reduction. Largest dataset accuracy loss: {r.worst_delta_pp:+.3f} pp.")
    recommendation=''
    conservative=summary[summary.variant=='fast30']
    if len(conservative) and conservative.iloc[0].datasets==55:
        r=conservative.iloc[0]
        recommendation=(f"Recommended conservative preset: **fast30** — retain all four operators at a nominal 30,000-feature budget (29,568 actual columns), use blocked ridge, buffered extraction, in-place cleaning/scaling, prediction batches of 256, and folded scaling. On this confirmation split it gives {r.accuracy*100:.3f}% mean accuracy ({r.delta_pp:+.3f} pp), {r.speedup:.2f}x full-runtime speedup and {r.median_rss_reduction_pct:.1f}% lower median paired peak RSS. Its mean accuracy change is an observed result; the bootstrap interval below describes its uncertainty. The 10k option is faster with an accuracy cost, while the hybrid has the highest observed mean accuracy and greater runtime.")
        atomic_choice['recommended_conservative_default']='fast30'
        base=summary[summary.variant=='baseline50'].iloc[0]
        recommendation+=(f" Classifier fitting alone takes {(r.classifier_fit_s/base.classifier_fit_s-1)*100:.1f}% longer, and median fitted-model pickle size is {(r.median_model_mib/base.median_model_mib-1)*100:.1f}% larger. Full fitting and prediction still improve because feature extraction is cheaper. Folded prediction retains an extra coefficient copy.")
    (report/'recommendation.json').write_text(json.dumps(atomic_choice,indent=2)+'\n')
    text=['# Best overall combination experiment','',
          recommendation,'',
          f"Frozen balanced candidate: **{choice['balanced']}**. Accuracy candidate: **{choice['accuracy_winner']}**. Fastest eligible candidate: **{choice['fastest_eligible']}**.",'',
          'The original finalist list was frozen before confirmation. The balanced rule allows at most 0.5 percentage points lower mean selection accuracy than the original 50k baseline, then minimizes the square root of the total runtime ratio times the median paired peak-RSS ratio. This is an explicit preference, not a claim that one model wins every metric.','',
          ('Conservative candidate: **'+extra['conservative']+'**, selected by a separate frozen rule: '+extra['rule']+'. This addition uses selection scores only and was frozen before that candidate was confirmed. The original choice remains unchanged in choice.json; the added rule and input hashes are in extra_choice.json.' if extra else 'No supplementary conservative candidate was selected.'),'',
          '## Selection on training data','',
          'Ten initial candidates were attempted on all 54 original selection datasets. Only their official TRAIN data were split into approximately 80/20 training/validation subsets: at least one item from each non-singleton class enters validation and at least one stays in training. Singleton classes remain in training. Small classes can therefore have a larger validation fraction. No official TEST labels were used to rank candidates. Each candidate uses the same subset and kernel seed for a dataset.','',
          markdown_table(select,fmt),'',
          ('A targeted eleventh configuration, **fast30**, replaces fused30\'s sklearn ridge with blocked ridge. It was motivated by selection-only ElectricDevices memory: fused30 reached 8.91 GiB, while fast50 used 6.47 GiB. All 54 selection predictions and alphas had to match fused30 before its confirmation was unlocked. Its separate immutable protocol and results are in `'+str(refinement)+'`.' if refinement else 'No fast30 refinement is included in this report.'),'',
          'Configuration names: baseline50 = original four operators, sklearn ridge, copying and full prediction; buffered50 = four operators, reused transform scratch, in-place cleaning/scaling, batches of 256; fused50 = buffered50 plus folded scaling at prediction. fused30/20/10 use the same changes at nominal budgets 30k/20k/10k. fast50 also uses the blocked ridge backend. pair50 retains LSPV+MPV with fast ridge, in-place operations, folded scaling and batching. hydra64 uses the official HYDRA transform with 64 groups and SparseScaler. hybrid20h32 combines buffered 20k four-operator MultiRocket with 32-group HYDRA; the two blocks use their respective scalers before a shared sklearn ridge classifier.','',
          '## Confirmation on archive resample 1','',
          f"Completed successful jobs: **{int((raw.status=='ok').sum())}/{len(protocol['split']['heldout'])*len(choice['finalists'])*protocol['repeats']}** across 55 dataset names and {protocol['repeats']} timing repetition(s). The finalist list was frozen from selection: {', '.join(choice['finalists'])}.",
          'This uses a different archive resample from the previous resample-0 experiment. It reuses the same underlying examples and dataset names, so it is confirmation on another split, not an entirely new independent dataset collection. Results were not used to change the frozen candidates.','',
          markdown_table(summary[['variant','datasets','accuracy','delta_pp','wins','ties','losses',
                                   'total_s','speedup','median_rss_reduction_pct','pareto']],fmt),'',
          *findings,'',
          '![Measured accuracy, runtime and memory tradeoffs](tradeoffs.png)','',
          'Runtime is complete fit plus prediction, summed across the paired datasets; startup, loading and warmup are excluded. Values above 1x are faster. Process peak RSS includes imports, inputs and warmup, and is supervised at 50ms intervals. It is different from feature-array memory. Pareto means no measured finalist is at least as good on accuracy, runtime and median peak RSS with a strict improvement in one.','',
          markdown_table(summary[['variant','fit_s','predict_s','train_transform_s','test_transform_s',
                                   'classifier_fit_s','classifier_predict_s','test_cases_per_s']],
                         {c:(lambda x:f'{x:.2f}') for c in ('fit_s','predict_s','train_transform_s',
                          'test_transform_s','classifier_fit_s','classifier_predict_s','test_cases_per_s')}),'',
          'All stage times are seconds, summed per-dataset medians. Classifier times include the relevant scaling work; total times also include kernel generation, differencing, cleaning and batching overhead. Medians of stage times do not have to add exactly to the median of their total. test_cases_per_s is aggregate batch throughput, not the latency of a single online request.','',
          markdown_table(summary[['variant','balanced_accuracy','macro_f1','median_features',
                                   'median_train_feature_mib','median_test_batch_mib',
                                   'median_rss_gib','max_rss_gib','median_model_mib']],fmt),'',
          'The median of paired RSS reductions is a different statistic from the reduction of the two median RSS values. Model size is the Python pickle size of the fitted model, including kernel parameters and coefficients, measured after the timed run. All raw records contain classifier/transform times, alpha and prediction hashes.','',
          f"Repeat stability: {int((stability.distinct_predictions==1).sum())}/{len(stability)} dataset/configuration pairs have identical predictions across their completed timing repetitions; {int((stability.distinct_alphas==1).sum())}/{len(stability)} have identical alpha. Full records are in repeat_stability.csv. Maximum RSS above is the largest observed successful-job peak, rather than a median.",'',
          'runtime_by_repeat.csv records the full fit/predict sums and dataset counts for each timing repetition. The main table sums per-dataset median times, which need not equal the median of the three run-level sums.','',
          '## Accuracy uncertainty','']
    for _,r in summary.iterrows():
        text.append(f"- {r.variant}: mean difference {r.delta_pp:+.3f} pp; dataset-bootstrap 95% interval [{r.delta_ci_low_pp:+.3f}, {r.delta_ci_high_pp:+.3f}] pp.")
    text+=['','Intervals resample dataset-level differences. They do not represent variation across 30 training resamples, and the selection accuracy threshold does not guarantee the same accuracy margin on confirmation.','',
           '## Dataset-level tradeoffs','',
           'Largest losses and gains for the balanced, conservative and accuracy candidates are shown below. Every dataset and all stage metrics are in per_dataset.csv.','']
    selected=list(dict.fromkeys([choice['balanced'],choice.get('refined_conservative',choice.get('conservative')),choice['accuracy_winner']]))
    for variant in selected:
        current=paired[paired.variant==variant]
        if not len(current): continue
        extreme=pd.concat([current.nsmallest(3,'delta_pp'),current.nlargest(3,'delta_pp')]).drop_duplicates('dataset')
        text+=[f'**{variant}**','',markdown_table(extreme[['dataset','baseline_accuracy','accuracy','delta_pp','speedup']],
               {**fmt,'baseline_accuracy':lambda x:f'{100*x:.3f}%'}),'']
    text+=[
           '## Numerical parity','']
    if len(parity):
        p=parity.groupby(['phase','variant','reference']).agg(jobs=('dataset','size'),alpha_matches=('alpha_equal','sum'),
                                                  minimum_prediction_agreement=('prediction_agreement','min')).reset_index()
        text += [markdown_table(p,{'minimum_prediction_agreement':lambda x:f'{100*x:.6f}%'}),'',
                 'Buffered scratch preserves the pooling definitions. Folded scaling is mathematically equivalent but changes floating-point rounding; the table gives observed prediction agreement rather than a universal bitwise guarantee.','']
    text+=['## Published reference','',
           f"On these 55 dataset names the published 50k MultiRocket accuracy is {paper.reindex(protocol['split']['heldout']).mean()*100:.3f}% on resample 1 and {means.reindex(protocol['split']['heldout']).mean()*100:.3f}% for its 30-resample mean. These are contextual references with different kernel seeds and software/hardware. Fresh paired baseline jobs are used for improvement claims.",'',
           '## ElectricDevices stress','']
    stress=records(output/'stress')
    if refinement:
        stress=pd.concat([stress,records(refinement/'stress')],ignore_index=True)
    if len(stress):
        stress.to_csv(report/'electricdevices.csv',index=False)
        columns=['variant','status','accuracy','total_s','peak_rss_bytes','features']
        text += [markdown_table(stress.reindex(columns=columns),{**fmt,
                   'accuracy':lambda x:'—' if pd.isna(x) else f'{100*x:.3f}%',
                   'total_s':lambda x:'—' if pd.isna(x) else f'{x:.2f}',
                   'features':lambda x:'—' if pd.isna(x) else f'{x:.0f}',
                   'peak_rss_bytes':lambda x:f'{x/1024**3:.2f} GiB'}),'',
                 'ElectricDevices belongs to the selection half and is excluded from held-out confirmation averages. Stress uses one attempt per variant. rss_limit means the 9 GiB experiment guard stopped the process; it does not establish an operating-system OOM.','']
        baseline_stress=stress[(stress.variant=='baseline50')&(stress.status=='ok')]
        fast_stress=stress[(stress.variant=='fast30')&(stress.status=='ok')]
        if len(baseline_stress) and len(fast_stress):
            b,r=baseline_stress.iloc[0],fast_stress.iloc[0]
            text += [f"The original baseline also completed this resample. fast30 lowers peak RSS from {b.peak_rss_bytes/1024**3:.2f} to {r.peak_rss_bytes/1024**3:.2f} GiB ({(1-r.peak_rss_bytes/b.peak_rss_bytes)*100:.1f}% lower), but total runtime rises from {b.total_s:.2f} to {r.total_s:.2f} seconds ({(r.total_s/b.total_s-1)*100:.1f}% longer). Accuracy changes from {b.accuracy*100:.3f}% to {r.accuracy*100:.3f}% ({(r.accuracy-b.accuracy)*100:+.3f} pp). The 55-dataset average gains do not apply to every dataset.",'']
    else:
        text+=['Stress jobs have not been run.','']
    routing_file=report/'backend_routing.json'
    if routing_file.exists():
        routing=json.loads(routing_file.read_text())
        text+=['## Optional backend routing analysis','',
               f"A selection-only size-based routing analysis chose {routing['threshold']:,} training cases as a switch: sklearn below this count, blocked ridge at or above it. The objective is the same time/RSS geometric score, with a maximum selection RSS of 8 GiB to leave 1 GiB below the experiment guard. This combines existing measured paths; an additional adaptive wrapper was not timed. All 55 confirmation datasets fall below this threshold and therefore use the measured fused30 path. ElectricDevices uses fast30. See backend_routing_selection.csv and backend_routing.json for every threshold and input hash. The fixed fast30 recommendation favors consistent memory headroom over this optional throughput policy.",'']
    text+=['## Reproduction and limits','',
           'Single-machine CPU measurements with four Numba/Torch threads, one BLAS thread, warm JIT and rotated execution order for the initial finalists. The fast30 refinement runs separately after them. HYDRA uses adaptive batches to bound convolution scratch; MultiRocket batches are 256. Every job runs in a fresh process. Exact source, dataset and resample-index hashes, package versions, choice hashes and raw predictions are retained in `'+str(output)+'`.','',
           'This is a finite candidate search; thread-count sweeps, GPU execution, iterative ridge solvers, learned feature pruning, energy use, and a full 30-resample replication were not measured. Blocked ridge still requires quadratic memory in training cases and a cubic eigendecomposition; it is not a streaming solver. Classifier alpha uses the default leave-one-out squared-error objective. HYDRA is the upstream GPL-3.0 implementation from https://github.com/angus924/hydra; its source and license are included under experiments/vendor.','',
           'The initial search and repetition commands are documented in README.md. Existing protocols reject source changes rather than mixing implementations. The repetition driver records where unchanged selection and first-repeat results were reused. All job failures, including selection and guard stops, are in failures.csv.','',
           'Generate this report with `.venv/bin/python -m experiments.summarize_combinations --output '+str(output)+(' --refinement-output '+str(refinement) if refinement else '')+'`.','']
    (report/'report.md').write_text('\n'.join(text))
    print(markdown_table(summary[['variant','datasets','accuracy','delta_pp','total_s','speedup','median_rss_reduction_pct']],fmt))
    print('Wrote '+str(report/'report.md'))


if __name__=='__main__':
    main()
