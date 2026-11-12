"""Write a lightweight local progress snapshot for the long-running review."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/review_revision/corrected_v1')
    parser.add_argument('--report-dir',default='reports/review_revision')
    args=parser.parse_args();root=Path(args.output);report=Path(args.report_dir)
    p=json.loads((root/'protocol.json').read_text());report.mkdir(parents=True,exist_ok=True)
    n=len(p['split']['heldout']);r=p['timing_repeats'];f=len(p['finalists'])
    expected=dict(ablation=len(p['ablation_datasets'])*len(p['ablation_configs'])*r,
      confirmation=n*(len(p['graph_configs'])+1+(r-1)*f),
      accuracy=n*(len(p['resamples'])-1)*len(p['accuracy_finalists']),stress=f,audit=len(p['audit_datasets']))
    counts={}
    for phase,total in expected.items():
        rows=[json.loads(file.read_text()) for file in (root/phase).glob('*.json')]
        counts[phase]=dict(saved=len(rows),expected=total,statuses=dict(Counter(row['status'] for row in rows)))
    complete=all(v['saved']==v['expected'] for v in counts.values())
    lines=['# Corrected review — '+('measurements complete' if complete else 'in progress'),'']
    lines+=['This is a saved progress snapshot. The final statistical/default decision is recorded in report.md and decision.json.','',
            '| Stage | Saved | Expected | Status counts |','| --- | ---: | ---: | --- |']
    for phase,v in counts.items():lines.append(f'| {phase} | {v["saved"]} | {v["expected"]} | {v["statuses"] or "pending"} |')
    lines+=['','Completed implementation: corrected L-1 convolution, byte-based ridge at 693 MiB, independent in-place/batching controls, direct convolution and threshold tests.','',
            'The ablation freezes the candidate before confirmation. No-loss is tested on three accuracy resamples at the predeclared -0.3 pp margin; timing repetitions are not extra accuracy samples. The old implementation and explicit legacy preset remain available.','']
    if (report/'report.md').exists():lines+=['[Open the comparison report](report.md).','']
    log=Path('/tmp/multirocket-corrected-review.log')
    if log.exists():lines+=['Recent run log:','','```',*log.read_text().splitlines()[-6:],'```','']
    (report/'progress.md').write_text('\n'.join(lines))
    (report/'progress.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts))


if __name__=='__main__':main()
