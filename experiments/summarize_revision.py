"""Review report with a predeclared -0.3 pp non-inferiority margin.

Dataset means across three accuracy resamples are the statistical units. The
three runtime repetitions of resample 1 never become extra accuracy samples.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from experiments.compare_optimizations import atomic_json,digest
from experiments.summarize_optimizations import markdown_table


def read_rows(path):
    return pd.DataFrame([json.loads(p.read_text()) for p in sorted(path.glob('*.json'))])


def paired_ni(difference_matrix,margin=-.3):
    """Rows are datasets, columns are train/test resamples, units are pp."""
    matrix=np.asarray(difference_matrix,dtype=float)
    if matrix.ndim!=2 or not np.isfinite(matrix).all(): raise ValueError('Complete dataset/resample pairs required')
    per_dataset=matrix.mean(axis=1)
    rng=np.random.default_rng(421)
    n,s=matrix.shape
    sample_d=rng.integers(0,n,size=(20000,n))
    bootstrap=per_dataset[sample_d].mean(axis=1)
    ci=np.quantile(bootstrap,[.025,.975])
    sample_s=rng.integers(0,s,size=(20000,n,s))
    hierarchical=matrix[sample_d[:,:,None],sample_s].mean(axis=(1,2))
    hci=np.quantile(hierarchical,[.025,.975])
    mean=float(per_dataset.mean()); se=0.0 if np.ptp(per_dataset)==0 else float(stats.sem(per_dataset))
    if se==0:
        tvalue=float('inf') if mean>margin else float('-inf')
        pvalue=0.0 if mean>margin else 1.0
        lower=mean
    else:
        result=stats.ttest_1samp(per_dataset,popmean=margin,alternative='greater')
        tvalue,pvalue=float(result.statistic),float(result.pvalue)
        lower=mean-float(stats.t.ppf(.95,n-1))*se
    return dict(datasets=n,resamples=s,mean_delta_pp=mean,ci_low_pp=float(ci[0]),ci_high_pp=float(ci[1]),
                hierarchical_ci_low_pp=float(hci[0]),hierarchical_ci_high_pp=float(hci[1]),
                margin_pp=margin,t_statistic=tvalue,p_value=pvalue,one_sided_t_lower_pp=lower,
                noninferior=bool(pvalue<.05 and ci[0]>margin))


def timing_summary(raw,baseline):
    ok=raw[raw.status=='ok']
    metrics=['accuracy','balanced_accuracy','macro_f1','total_s','fit_s','predict_s',
             'classifier_fit_s','classifier_predict_s','train_transform_s','test_transform_s',
             'peak_rss_bytes','train_feature_bytes','max_prediction_feature_bytes','model_bytes','features','n_train','n_test']
    m=ok.groupby(['dataset','variant'])[metrics].median().reset_index()
    b=m[m.variant==baseline].set_index('dataset')
    summary=[];pairs=[]
    for variant in m.variant.unique():
        v=m[m.variant==variant].set_index('dataset');names=b.index.intersection(v.index)
        if not len(names):continue
        bv,vv=b.loc[names],v.loc[names]
        delta=100*(vv.accuracy-bv.accuracy)
        row=dict(variant=variant,datasets=len(names),accuracy=float(vv.accuracy.mean()),
          delta_pp=float(delta.mean()),wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),
          balanced_accuracy=float(vv.balanced_accuracy.mean()),macro_f1=float(vv.macro_f1.mean()),
          total_s=float(vv.total_s.sum()),fit_s=float(vv.fit_s.sum()),predict_s=float(vv.predict_s.sum()),
          speedup=float(bv.total_s.sum()/vv.total_s.sum()),
          rss_reduction_pct=float((1-vv.peak_rss_bytes/bv.peak_rss_bytes).median()*100),
          median_rss_gib=float(vv.peak_rss_bytes.median()/1024**3),
          max_rss_gib=float(ok[(ok.variant==variant)&ok.dataset.isin(names)].peak_rss_bytes.max()/1024**3),
          model_mib=float(vv.model_bytes.median()/1024**2),
          train_features_mib=float(vv.train_feature_bytes.median()/1024**2),
          prediction_features_mib=float(vv.max_prediction_feature_bytes.median()/1024**2),
          features=float(vv.features.median()),classifier_fit_s=float(vv.classifier_fit_s.sum()),
          classifier_predict_s=float(vv.classifier_predict_s.sum()),train_transform_s=float(vv.train_transform_s.sum()),
          test_transform_s=float(vv.test_transform_s.sum()),worst_delta_pp=float(delta.min()),best_delta_pp=float(delta.max()))
        summary.append(row)
        for d in names:
            pairs.append({**vv.loc[d].to_dict(),'dataset':d,'variant':variant,
                          'baseline_accuracy':float(bv.loc[d,'accuracy']),'delta_pp':float(delta.loc[d]),
                          'speedup':float(bv.loc[d,'total_s']/vv.loc[d,'total_s'])})
    return pd.DataFrame(summary),pd.DataFrame(pairs),m


def analyze_ablation(raw):
    summary,pairs,medians=timing_summary(raw,'baseline30')
    parity=[]
    b=raw[(raw.variant=='baseline30')&(raw.status=='ok')].set_index(['dataset','repeat'])
    for variant in summary.variant:
        if variant=='feature50':continue
        v=raw[(raw.variant==variant)&(raw.status=='ok')].set_index(['dataset','repeat'])
        for idx in b.index.intersection(v.index):
            parity.append(dict(dataset=idx[0],repeat=idx[1],variant=variant,
                               alpha_equal=bool(v.loc[idx,'alpha']==b.loc[idx,'alpha']),
                               predictions_equal=bool(v.loc[idx,'prediction_hash']==b.loc[idx,'prediction_hash'])))
    return summary,pairs,medians,pd.DataFrame(parity)


def aggregate_accuracy(raw):
    valid=raw[raw.status=='ok']
    return valid.groupby(['dataset','variant','resample'])[['accuracy','balanced_accuracy','macro_f1']].mean().reset_index()


def accuracy_inference(acc,variants,datasets,resamples,margin):
    results=[];differences=[]
    for variant in variants:
        pivot=acc[acc.variant.isin(['baseline50',variant])].pivot(index=['dataset','resample'],columns='variant',values='accuracy')
        if variant not in pivot or 'baseline50' not in pivot:continue
        delta=(pivot[variant]-pivot['baseline50'])*100
        matrix=delta.unstack('resample').reindex(index=datasets,columns=resamples)
        if matrix.isna().any().any():continue
        result=paired_ni(matrix.to_numpy(),margin);result['variant']=variant
        result['accuracy']=float(acc[(acc.variant==variant)&acc['resample'].isin(resamples)].accuracy.mean())
        result['baseline_accuracy']=float(acc[(acc.variant=='baseline50')&acc['resample'].isin(resamples)].accuracy.mean())
        results.append(result)
        for dataset,row in matrix.iterrows():
            differences.append(dict(dataset=dataset,variant=variant,mean_delta_pp=float(row.mean()),
                                    **{f'resample_{s}_delta_pp':float(row[s]) for s in matrix.columns}))
    return pd.DataFrame(results),pd.DataFrame(differences)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/review_revision/corrected_v1')
    parser.add_argument('--report-dir',default='reports/review_revision')
    args=parser.parse_args();out,report=Path(args.output),Path(args.report_dir);report.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((out/'protocol.json').read_text())
    choice=json.loads((out/'bundle_choice.json').read_text())
    ab=read_rows(out/'ablation');confirmation=read_rows(out/'confirmation');accuracy=read_rows(out/'accuracy')
    stress=read_rows(out/'stress');audit=read_rows(out/'audit')
    ab_summary,ab_pairs,_,ab_parity=analyze_ablation(ab)
    ab_summary.to_csv(report/'ablation_summary.csv',index=False);ab_pairs.to_csv(report/'ablation_per_dataset.csv',index=False)
    ab_parity.to_csv(report/'ablation_parity.csv',index=False);ab.to_csv(report/'ablation_raw.csv',index=False)
    if not len(confirmation):raise ValueError('No confirmation measurements')
    primary=confirmation[confirmation.variant.isin(protocol['finalists'])]
    summary,pairs,medians=timing_summary(primary,'baseline50')
    summary.to_csv(report/'finalists.csv',index=False);pairs.to_csv(report/'per_dataset.csv',index=False)
    medians.to_csv(report/'stage_times_and_memory.csv',index=False)
    confirmation.to_csv(report/'confirmation_raw.csv',index=False)
    stability=confirmation[confirmation.status=='ok'].groupby(['dataset','variant']).agg(
        completed_repeats=('repeat','size'),distinct_predictions=('prediction_hash','nunique'),
        distinct_alphas=('alpha','nunique')).reset_index()
    stability.to_csv(report/'repeat_stability.csv',index=False)
    # All original eleven plus auto30 share the same run number for the chart.
    chart_raw=confirmation[(confirmation.repeat==0)&(confirmation.status=='ok')]
    graph_variants=list(protocol['graph_configs'])+['auto30']
    counts=chart_raw.groupby('dataset').variant.nunique()
    common=counts[counts==len(graph_variants)].index
    chart,chart_pairs,_=timing_summary(chart_raw[chart_raw.dataset.isin(common)],'baseline50')
    chart.to_csv(report/'all_configurations.csv',index=False)
    # Accuracy is aggregated once per dataset and resample, never per timing repeat.
    combined=pd.concat([confirmation,accuracy],ignore_index=True) if len(accuracy) else confirmation.copy()
    valid=combined[combined.status=='ok']
    acc=aggregate_accuracy(combined)
    acc.to_csv(report/'accuracy_by_resample.csv',index=False)
    sensitivity=[]
    for split in protocol['resamples']:
        current=valid[(valid['resample']==split)&valid.variant.isin(protocol['accuracy_finalists'])]
        if not len(current):continue
        coverage=current.groupby('variant').dataset.nunique()
        if len(coverage)!=3 or not (coverage==55).all():continue
        table,_,_=timing_summary(current,'baseline50')
        table['resample']=split
        sensitivity.append(table)
    runtime_sensitivity=pd.concat(sensitivity,ignore_index=True) if sensitivity else pd.DataFrame()
    runtime_sensitivity.to_csv(report/'runtime_by_resample.csv',index=False)
    ni,delta_rows=accuracy_inference(acc,('fast30','auto30'),protocol['split']['heldout'],protocol['resamples'],protocol['noninferiority_margin_pp'])
    ni.to_csv(report/'noninferiority.csv',index=False)
    delta_rows.to_csv(report/'paired_resample_differences.csv',index=False)
    if len(stress):stress.to_csv(report/'electricdevices.csv',index=False)
    if len(audit):audit.to_csv(report/'legacy_replay.csv',index=False)
    # All numerical parity comparisons retain the representation and split.
    parity=[]
    for phase,raw in [('confirmation',confirmation),('accuracy',accuracy)]:
        if not len(raw):continue
        b=raw[(raw.variant=='fast30')&(raw.status=='ok')].set_index(['dataset','resample','repeat'])
        v=raw[(raw.variant=='auto30')&(raw.status=='ok')].set_index(['dataset','resample','repeat'])
        for idx in b.index.intersection(v.index):
            pred=np.load(out/phase/f'{idx[0]}__auto30__s{idx[1]}__r{idx[2]}.npy')
            ref=np.load(out/phase/f'{idx[0]}__fast30__s{idx[1]}__r{idx[2]}.npy')
            parity.append(dict(phase=phase,dataset=idx[0],resample=idx[1],repeat=idx[2],
                               alpha_equal=bool(b.loc[idx,'alpha']==v.loc[idx,'alpha']),agreement=float(np.mean(pred==ref))))
    parity=pd.DataFrame(parity);parity.to_csv(report/'ridge_parity.csv',index=False)
    failures=[]
    for phase in ('ablation','confirmation','accuracy','stress','audit'):
        raw=read_rows(out/phase)
        if len(raw):
            failed=raw[raw.status!='ok'].copy();failed['phase']=phase;failures.extend(failed.to_dict('records'))
    pd.DataFrame(failures).to_csv(report/'failures.csv',index=False)
    # Strict, predeclared adoption gate. No silent fallback to incomplete pairs.
    decision=dict(adopt=False,reasons=[],previous_configuration={**protocol['graph_configs']['fast30'],'first_difference_fix':False},
                  corrected_previous_configuration={**protocol['graph_configs']['fast30'],'first_difference_fix':True},
                  candidate_configuration=choice['configuration'],margin_pp=protocol['noninferiority_margin_pp'])
    decision['protocol_sha256']=digest(out/'protocol.json')
    decision['analysis_source_sha256']=digest(__file__)
    if (out/'analysis_plan.json').exists():
        decision['analysis_plan_sha256']=digest(out/'analysis_plan.json')
        (report/'analysis_plan.json').write_bytes((out/'analysis_plan.json').read_bytes())
    new=summary[summary.variant=='auto30'];previous=summary[summary.variant=='fast30'];new_ni=ni[ni.variant=='auto30'] if len(ni) else ni
    if len(new) and len(previous) and len(new_ni):
        n,p,z=new.iloc[0],previous.iloc[0],new_ni.iloc[0]
        decision['runtime_gain_vs_previous']=float(p.total_s/n.total_s)
        # Per-dataset paired RSS, rather than a ratio of unrelated medians.
        nrows=medians[medians.variant=='auto30'].set_index('dataset');prows=medians[medians.variant=='fast30'].set_index('dataset')
        names=nrows.index.intersection(prows.index)
        ratio=float((nrows.loc[names,'peak_rss_bytes']/prows.loc[names,'peak_rss_bytes']).median())
        decision['paired_rss_ratio_vs_previous']=ratio
        tests={
          'complete_six_finalists':len(primary[primary.status=='ok'])==55*6*3,
          'noninferior':bool(z.noninferior),
          'runtime_gain_at_least_3pct':bool(p.total_s/n.total_s>=1.03),
          'rss_within_20pct_of_previous':bool(ratio<=1.20),
          'electricdevices_success':bool(len(stress) and ((stress.variant=='auto30')&(stress.status=='ok')&(stress.peak_rss_bytes<9*1024**3)).any()),
          'alpha_parity':bool(len(parity)==55*(3+2) and parity.alpha_equal.all()),
          'legacy_replay_intact':bool(len(audit)==5 and (audit.status=='ok').all() and audit.legacy_reference_prediction_equal.all() and audit.legacy_reference_alpha_equal.all()),
        }
        decision['gates']=tests;decision['adopt']=all(tests.values())
        decision['reasons']=[k for k,v in tests.items() if not v]
    else:decision['reasons']=['Accuracy confirmation or finalist timings incomplete']
    expected_confirmation=55*(len(protocol['graph_configs'])+1+2*len(protocol['finalists']))
    complete_attempts=(len(confirmation)==expected_confirmation and len(accuracy)==330 and len(stress)==6 and len(audit)==5)
    decision['evaluation_complete']=bool(complete_attempts)
    decision['status']='pending' if not complete_attempts else ('adopt' if decision['adopt'] else 'retain_previous')
    if not complete_attempts:
        decision['adopt']=False
        if 'Remaining experiment stages pending' not in decision['reasons']:
            decision['reasons'].append('Remaining experiment stages pending')
    atomic_json(report/'decision.json',decision)
    # Publication/export artifacts use standard plotting tools.
    os.environ.setdefault('MPLCONFIGDIR','/tmp/multirocket-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    best_accuracy=-np.inf;frontier=[]
    for _,r in chart.sort_values('speedup',ascending=False).iterrows():
        if r.accuracy>best_accuracy:
            frontier.append((r.speedup,r.accuracy*100));best_accuracy=r.accuracy
    if frontier:
        f=np.array(sorted(frontier));ax.plot(f[:,0],f[:,1],color='.55',lw=1.2,ls='--',zorder=1)
    for i,(_,r) in enumerate(chart.sort_values('speedup').iterrows()):
        marker='*' if r.variant=='auto30' else 'o'
        point=ax.scatter(r.speedup,r.accuracy*100,s=180 if marker=='*' else 65,marker=marker,
                         color=plt.get_cmap('tab20')(i),label=r.variant,zorder=3)
        point.set_gid(r.variant)
    ax.axvline(1,color='.6',ls='--',lw=1)
    baseline_point=chart[chart.variant=='baseline50']
    if len(baseline_point):ax.axhline(float(baseline_point.iloc[0].accuracy)*100,color='.7',ls=':',lw=1)
    ax.grid(alpha=.2)
    ax.set_xlabel('Complete fit + prediction speedup (higher is better)');ax.set_ylabel('Mean test accuracy (%)')
    ax.set_title(f'Corrected kernels: all 11 original configurations + size-based ridge\n{len(common)} common UCR datasets, resample 1, uniform first timing run')
    ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=9,title='Configuration')
    fig.savefig(report/'tradeoffs_all.png',dpi=200);fig.savefig(report/'tradeoffs_all.svg');plt.close(fig)
    import xml.etree.ElementTree as ET
    svg=ET.parse(report/'tradeoffs_all.svg');namespace='{http://www.w3.org/2000/svg}'
    by_name=chart.set_index('variant')
    for element in svg.getroot().iter():
        name=element.get('id')
        if name in by_name.index:
            row=by_name.loc[name];tip=ET.Element(namespace+'title')
            tip.text=f'{name}: accuracy {row.accuracy*100:.3f}%; speedup {row.speedup:.3f}x; RSS reduction {row.rss_reduction_pct:.1f}%'
            element.insert(0,tip)
    svg.write(report/'tradeoffs_all.svg',encoding='utf-8',xml_declaration=True)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    ordered=ab_summary.sort_values('speedup')
    axes[0].barh(ordered.variant,ordered.speedup,color=['#2b8cbe' if v>=1 else '#d95f0e' for v in ordered.speedup])
    axes[0].axvline(1,color='.4',ls='--');axes[0].set_xlabel('Speedup versus plain 30k sklearn')
    axes[1].barh(ordered.variant,ordered.rss_reduction_pct,color='#2b8cbe')
    axes[1].axvline(0,color='.4',ls='--');axes[1].set_xlabel('Median paired peak RSS reduction (%)')
    fig.suptitle('Bundle ablation: one change at a time, plus full bundle controls')
    fig.savefig(report/'ablation.png',dpi=180);plt.close(fig)
    fmt=dict(accuracy=lambda x:f'{100*x:.3f}%',baseline_accuracy=lambda x:f'{100*x:.3f}%',
             balanced_accuracy=lambda x:f'{100*x:.3f}%',macro_f1=lambda x:f'{100*x:.3f}%',
             delta_pp=lambda x:f'{x:+.3f}',speedup=lambda x:f'{x:.2f}x',
             rss_reduction_pct=lambda x:f'{x:.1f}%',total_s=lambda x:f'{x:.2f}',
             fit_s=lambda x:f'{x:.2f}',predict_s=lambda x:f'{x:.2f}',
             model_mib=lambda x:f'{x:.2f}',train_features_mib=lambda x:f'{x:.2f}',
             prediction_features_mib=lambda x:f'{x:.2f}',median_rss_gib=lambda x:f'{x:.2f}',max_rss_gib=lambda x:f'{x:.2f}')
    headline='Evaluation is incomplete; the previous preset remains available.'
    if len(new_ni):
        z=new_ni.iloc[0]
        if bool(z.noninferior):
            wording='Accuracy unchanged within' if z.ci_low_pp<=0<=z.ci_high_pp else 'Accuracy non-inferior within'
            headline=f"{wording} the predeclared 0.3-point non-inferiority margin ({z.mean_delta_pp:+.3f} pp; dataset-bootstrap 95% CI [{z.ci_low_pp:+.3f}, {z.ci_high_pp:+.3f}])."
        else:
            headline=f"No-loss is not established: {z.mean_delta_pp:+.3f} pp; dataset-bootstrap 95% CI [{z.ci_low_pp:+.3f}, {z.ci_high_pp:+.3f}], margin -0.300 pp."
    text=['# Corrected MultiRocket review','',headline,'',
      '**Decision: '+('pending — the experiment is still running; the previous preset remains the default.' if not complete_attempts else ('adopt the size-based 30k preset.' if decision['adopt'] else 'retain the previous backend preset; adoption gates did not all pass.'))+'**',
      ('Failed/incomplete gates: '+', '.join(decision['reasons'])+'.' if decision['reasons'] else 'All predeclared timing, memory, non-inferiority, parity and ElectricDevices gates passed.'),'',
      '## First-difference audit','',
      'Earlier reports used legacy first-difference features. The difference array has length L-1 but the original transform used L when accumulating the left shifts. The corrected transforms use L-1. Raw-series features and pooling definitions are unchanged. Independent direct-correlation tests cover multiple lengths/dilations, constant series, all operators and pairs; buffered and unbuffered corrected features are also checked on float data. All new comparisons below, including the 50k reference and HYDRA hybrid, use corrected kernels. Standalone HYDRA has no MultiRocket difference kernel.','',
      'The original transform and buffered implementation remain available behind first_difference_fix=False / --legacy-first-difference. The prior report is archived in reports/combination_search; the prior source snapshot is in output/combination_search/previous_implementation. Legacy replay results are in legacy_replay.csv. Old and corrected metrics are not pooled.','',
      '## Size-based ridge','',
      'The switch uses estimated training feature bytes = training rows × actual feature columns × 4. At 30k nominal there are 29,568 columns. Below 693 MiB use sklearn ridge; at or above 693 MiB use blocked ridge. This is the previous selection-derived 6,144-row threshold expressed in bytes, not retuned on new confirmation labels. The selected backend is logged for every job.','',
      '## Six-finalist confirmation','',
      f"{int((primary.status=='ok').sum())}/{55*6*3} successful finalist timing jobs on resample 1. Three timing repetitions reuse the same split and seed. They are not three accuracy resamples. The table sums per-dataset median times; stage medians need not sum exactly to the median of their total.",'',
      markdown_table(summary[['variant','datasets','accuracy','delta_pp','total_s','speedup','rss_reduction_pct']],fmt),'',
      markdown_table(summary[['variant','wins','ties','losses','worst_delta_pp','best_delta_pp']],
                     {**fmt,'worst_delta_pp':lambda x:f'{x:+.3f}','best_delta_pp':lambda x:f'{x:+.3f}'}),'',
      'Wins/ties/losses and extreme changes above compare each dataset to the corrected 50k reference on resample 1. The non-inferiority statement below concerns the mean paired difference, not every dataset.','',
      markdown_table(summary[['variant','fit_s','predict_s','balanced_accuracy','macro_f1','median_rss_gib','max_rss_gib','model_mib']],fmt),'',
      'The frozen bundle choice and all flags are in decision.json and the experiment bundle_choice.json. Full per-dataset results, training/prediction feature buffers, stage times and model sizes are in the accompanying CSVs.','',
      f"Timing-repeat stability: {int((stability.distinct_predictions==1).sum())}/{len(stability)} dataset/configuration pairs have one prediction hash; {int((stability.distinct_alphas==1).sum())}/{len(stability)} have one alpha. Exact records are in repeat_stability.csv.",'',
      '## Accuracy and non-inferiority','',
      'Before these runs, the acceptable mean loss was set to 0.3 percentage points. The primary statistical units are the 55 datasets: average each paired accuracy difference over resamples 1, 2 and 3, then test those dataset means against -0.3 pp. We require a one-sided paired-difference t-test at alpha .05 and a two-sided dataset-bootstrap 95% lower bound above -0.3 pp. A two-level dataset/resample bootstrap is also reported as sensitivity analysis. These are benchmark-level statements, not guarantees for individual datasets or external populations.','']
    nfmt={c:(lambda x:f'{x:+.3f}') for c in ('mean_delta_pp','ci_low_pp','ci_high_pp','hierarchical_ci_low_pp','hierarchical_ci_high_pp','one_sided_t_lower_pp')}
    nfmt['p_value']=lambda x:f'{x:.4g}'
    if len(ni):text += [markdown_table(ni[['variant','datasets','resamples','mean_delta_pp','ci_low_pp','ci_high_pp','p_value','noninferior']],nfmt),'',
       markdown_table(ni[['variant','hierarchical_ci_low_pp','hierarchical_ci_high_pp','one_sided_t_lower_pp']],nfmt),'']
    text+=['Resamples reuse underlying examples, and UCR datasets can be related. We do not treat all 165 dataset/resample pairs as independent observations. “Unchanged” means no loss beyond the predeclared margin under this benchmark analysis; it does not establish equality. No claim of an accuracy gain is made. The paired-difference test uses [SciPy ttest_1samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).','',
      'Runtime sensitivity across complete resamples is retained in runtime_by_resample.csv. Resample 1 has three timing repetitions; resamples 2 and 3 have one timing each and are secondary timing evidence. They do not replace the predeclared resample-1 default-adoption comparison.','',
      '## Bundle ablation','',
      'This quick ablation uses 12 workload-selected datasets from the selection half, internal stratified validation, caps of 512 training and 512 validation cases, and three timing repetitions. It is for component cost and parity, not for claiming archive accuracy. Every single-change row starts from plain corrected 30k sklearn. feature50 provides the feature-budget control. full_auto and full_no_fold test the complete bundle.','',
      'On these capped cases routing_only selects the same sklearn backend as baseline30. Its apparent runtime difference therefore provides a useful noise control: small 1–3% changes are not strong evidence of a component speedup. In particular, batching cannot cause a training-fit speedup; variation in that column is measurement noise. Component gains are not additive. Large-set memory effects are checked separately on ElectricDevices.','',
      markdown_table(ab_summary[['variant','datasets','speedup','rss_reduction_pct','model_mib','delta_pp']],fmt),'',
      '![Bundle ablation](ablation.png)','',
      f"Folding removed from the candidate: **{choice['folding_removed']}**. No-fold/full ratios: runtime {choice['time_ratio_no_fold']:.4f}, paired RSS {choice['rss_ratio_no_fold']:.4f}, model size {choice['model_ratio_no_fold']:.4f}. Exact prediction/alpha parity across ablation jobs: {choice['exact_ablation_parity']}. The choice rule was frozen before confirmation. Buffered extraction, in-place processing and chunking remain enabled; their isolated costs are visible above.",'',
      '## All configurations on the tradeoff curve','',
      f"All eleven original configurations plus the new auto30 candidate appear on the same corrected resample-1 comparison. The chart uses the same first timing run for every configuration and {len(common)} common successful datasets. Six selected finalists additionally have three timing repetitions in the primary table. Non-finalist chart points are descriptive, with one timing run. No chart point is silently replaced with a result from a different kernel version or dataset subset.",'',
      '![All configuration tradeoffs](tradeoffs_all.png)','',
      'The dashed line is the observed accuracy/runtime Pareto frontier. Every point remains visible, including 10k and HYDRA. The [SVG figure](tradeoffs_all.svg) includes native hover titles with point values.','',
      markdown_table(chart[['variant','datasets','accuracy','delta_pp','speedup','rss_reduction_pct']],fmt),'',
      '## ElectricDevices stress','']
    if len(stress):
        text += [markdown_table(stress.reindex(columns=['variant','status','accuracy','total_s','peak_rss_bytes','selected_backend']),
                  {**fmt,'accuracy':lambda x:'—' if pd.isna(x) else f'{100*x:.3f}%',
                   'total_s':lambda x:'—' if pd.isna(x) else f'{x:.2f}',
                   'peak_rss_bytes':lambda x:f'{x/1024**3:.2f} GiB'}),'']
    text+=['ElectricDevices is in the selection half and excluded from the 55-dataset aggregates. Stress has one attempt per variant. rss_limit is a stop by the 9 GiB polling guard, not proof of an operating-system OOM. All stops remain in failures.csv.','',
      '## Reproduction and limits','',
      'Run `.venv/bin/python -m experiments.review_revision --stage all`, then `.venv/bin/python -m experiments.summarize_revision`. A frozen protocol, source snapshots, dataset/index hashes, model configurations, predictions and per-job logs remain under `'+str(out)+'`. Four Numba/Torch threads and one BLAS thread are used. Imports, data loading and warmup are excluded from timings; process RSS includes them.','',
      'This is three accuracy resamples, not the paper’s 30-resample replication. The new “held-out” dataset names have been inspected during earlier development; these are paired confirmation measurements, not a newly independent collection. Blocked ridge still needs quadratic memory and a cubic eigendecomposition in training cases. No energy or GPU measurements were made.','']
    (report/'report.md').write_text('\n'.join(text))
    print(headline);print('Adopt:',decision['adopt'],'reasons:',decision['reasons'])
    print(markdown_table(summary[['variant','accuracy','speedup','rss_reduction_pct']],fmt))
    print('Wrote '+str(report/'report.md'))


if __name__=='__main__': main()
