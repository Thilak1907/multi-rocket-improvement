#!/usr/bin/env python3
"""Compare the manuscript's four results tables with the retained measurements.

This audits reported values, not the validity of every scientific claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table_rows(text, label):
    blocks = re.findall(r'\\begin\{table\}.*?\\end\{table\}', text, re.S)
    block = next(b for b in blocks if '\\label{' + label + '}' in b)
    body = block.split('\\midrule', 1)[1].split('\\bottomrule', 1)[0]
    rows = []
    for line in body.splitlines():
        if '&' in line:
            cells = line.rstrip().removesuffix('\\\\').split('&')
            # Result cells contain one numeric value each (or -- for baseline RSS).
            rows.append([re.search(r'[+-]?\d+(?:\.\d+)?', cell).group()
                         if re.search(r'[+-]?\d+(?:\.\d+)?', cell) else None
                         for cell in cells[1:]])
    return rows


def compare_tables(text):
    finalists = pd.read_csv(ROOT/'reports/review_revision/finalists.csv').set_index('variant')
    accuracy = pd.read_csv(ROOT/'reports/review_revision/accuracy_by_resample.csv')
    metrics = accuracy[accuracy['resample'].isin([1,2,3])].groupby('variant')[['accuracy','balanced_accuracy','macro_f1']].mean()
    ensemble = pd.read_csv(ROOT/'reports/accuracy_search/confirmation_summary.csv').set_index('variant')
    choice = json.loads((ROOT/'reports/accuracy_search/decision.json').read_text())
    expected = {}
    expected['tab:finalists'] = [[100*finalists.loc[v,'accuracy'],finalists.loc[v,'speedup'],
                                 None if v=='baseline50' else finalists.loc[v,'rss_reduction_pct']]
                                for v in ['baseline50','fast30','auto30','fused30','fused10','hybrid20h32']]
    expected['tab:metrics'] = [list(100*metrics.loc[v]) for v in ['baseline50','fast30']]
    expected['tab:stages'] = []
    for col in ['fit_s','predict_s','classifier_fit_s','classifier_predict_s','train_transform_s','test_transform_s']:
        a,b=finalists.loc[['baseline50','fast30'],col]
        expected['tab:stages'].append([a,b,100*(1-b/a)])
    expected['tab:ensemble'] = []
    for metric in ['accuracy','balanced_accuracy','macro_f1']:
        expected['tab:ensemble'].append(list(100*ensemble.loc[['baseline','coarse'],metric]))
    expected['tab:ensemble'] += [[1.,choice['runtime_ratio']],list(ensemble.loc[['baseline','coarse'],'model_bytes']/1024**2)]
    checks=[]
    for label, rows in expected.items():
        actual=table_rows(text,label)
        if len(actual)!=len(rows):
            raise ValueError(f'{label}: expected {len(rows)} rows, found {len(actual)}')
        for i,(got,want) in enumerate(zip(actual,rows),1):
            if len(got)!=len(want):raise ValueError(f'{label}, row {i}: column count differs')
            for j,(shown,value) in enumerate(zip(got,want),1):
                if value is None:
                    ok=shown is None
                elif shown is None:
                    ok=False
                else:
                    decimals=len(shown.split('.')[1]) if '.' in shown else 0
                    ok=abs(float(shown)-value)<=.5*10**(-decimals)+1e-9
                checks.append(dict(table=label,row=i,column=j,manuscript=shown,recorded=value,match=bool(ok)))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manuscript',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'output/manuscript_audit')
    args=parser.parse_args()
    text=args.manuscript.read_text()
    checks=compare_tables(text)
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(checks).to_csv(args.out/'table_checks.csv',index=False)
    ni=pd.read_csv(ROOT/'reports/review_revision/noninferiority.csv').set_index('variant').loc['fast30']
    summary=dict(manuscript_sha256=hashlib.sha256(args.manuscript.read_bytes()).hexdigest(),
                 tables=4,numeric_cells=len(checks),all_cells_match=all(c['match'] for c in checks),
                 accuracy_delta_pp=ni.mean_delta_pp,ci_low_pp=ni.ci_low_pp,ci_high_pp=ni.ci_high_pp,
                 noninferiority_p=ni.p_value,
                 note='A matching numeric table does not validate unsupported prose claims or external references.')
    (args.out/'numeric_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not summary['all_cells_match']:
        print(pd.DataFrame([c for c in checks if not c['match']]).to_string(index=False))
        raise SystemExit(1)


if __name__=='__main__':main()
