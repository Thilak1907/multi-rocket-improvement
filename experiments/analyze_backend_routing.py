"""Analyze a training-size routing option using selection measurements only.

This combines already measured backend paths; it is not an additional timed
estimator. A one-GiB margin below the nine-GiB guard is required on selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.compare_optimizations import atomic_json, digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='output/combination_search/20261111')
    parser.add_argument('--refinement',default='output/combination_search/20261111_fast30')
    parser.add_argument('--report-dir',default='reports/combination_search')
    args=parser.parse_args()
    source,refinement,report=Path(args.source),Path(args.refinement),Path(args.report_dir)
    protocol=json.loads((source/'protocol.json').read_text())
    hashes={}
    rows={variant:[] for variant in ('baseline50','fused30','fast30')}
    for dataset in protocol['split']['selection']:
        for variant in rows:
            path=(refinement if variant=='fast30' else source)/'selection'/f'{dataset}__{variant}__0.json'
            row=json.loads(path.read_text())
            if row['status']!='ok': raise ValueError('Incomplete selection '+str(path))
            rows[variant].append(row)
            hashes[str(path)]=digest(path)
        if rows['fused30'][-1]['prediction_hash']!=rows['fast30'][-1]['prediction_hash']:
            raise ValueError('Backend prediction parity failed')
        if rows['fused30'][-1]['alpha']!=rows['fast30'][-1]['alpha']:
            raise ValueError('Backend alpha parity failed')
    b,s,f=(pd.DataFrame(rows[v]).set_index('dataset') for v in ('baseline50','fused30','fast30'))
    candidates=[]
    for threshold in (0,1024,2048,4096,6144,8192,16384):
        route=s.n_train>=threshold
        v=s.copy()
        v.loc[route]=f.loc[v.index[route]]
        tr=float(v.total_s.sum()/b.total_s.sum())
        rr=float((v.peak_rss_bytes/b.peak_rss_bytes).median())
        candidates.append(dict(threshold=threshold,fast_datasets=int(route.sum()),
                               max_rss_gib=float(v.peak_rss_bytes.max()/1024**3),
                               time_ratio=tr,rss_ratio=rr,cost_score=float(np.sqrt(tr*rr))))
    table=pd.DataFrame(candidates)
    eligible=table[table.max_rss_gib<=8]
    if not len(eligible): raise ValueError('No routing profile has the requested headroom')
    best=eligible.sort_values(['cost_score','threshold']).iloc[0]
    value=dict(threshold=int(best.threshold),max_selection_rss_gib=float(best.max_rss_gib),
               rule='Minimum sqrt(time ratio * median paired peak RSS ratio), with selection maximum RSS <= 8 GiB',
               profile='fused30 below threshold; fast30 at or above threshold',
               separately_timed=False,selection_files=hashes,analysis_source_sha256=digest(__file__))
    report.mkdir(parents=True,exist_ok=True)
    target=report/'backend_routing.json'
    if target.exists() and json.loads(target.read_text())!=value:
        raise ValueError('Previously frozen routing analysis differs')
    atomic_json(target,value)
    table.to_csv(report/'backend_routing_selection.csv',index=False)
    print(table.to_string(index=False))
    print('Selected routing threshold:',int(best.threshold))


if __name__=='__main__':
    main()
