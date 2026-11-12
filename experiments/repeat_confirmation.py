"""Reuse frozen selection and repeat-0 jobs; add two timing repetitions.

The new protocol explicitly records where unchanged measurements came from.
Selection is never rerun or revised using confirmation outcomes.
"""
import argparse
import json
from pathlib import Path
import shutil

from experiments.compare_optimizations import atomic_json, digest
from experiments.search_combinations import SOURCES, ROOT, run_job


def freeze_conservative(source):
    """A separate selection-only rule; never read confirmation outcomes."""
    import pandas as pd
    protocol=json.loads((source/'protocol.json').read_text())
    table=pd.read_csv(source/'selection_summary.csv')
    candidates=table[(table.delta_pp>=0)&(table.datasets==len(protocol['split']['selection']))&
                     (table.successful==len(protocol['split']['selection']))]
    variant=candidates.sort_values(['time_ratio','variant']).iloc[0].variant
    value=dict(conservative=variant,
               rule='Fastest complete selection candidate with no observed mean validation accuracy loss',
               selection_summary_sha256=digest(source/'selection_summary.csv'),
               original_choice_sha256=digest(source/'choice.json'))
    file=source/'extra_choice.json'
    if file.exists() and json.loads(file.read_text())!=value:
        raise ValueError('Conservative choice differs from its frozen selection inputs')
    if not file.exists(): atomic_json(file,value)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='output/combination_search/20261111')
    parser.add_argument('--output',default='output/combination_search/20261111_repeats3')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--stage',choices=('confirmation','stress'),default='confirmation')
    parser.add_argument('--stress-repeats',type=int,default=1)
    args=parser.parse_args()
    source,target=Path(args.source),Path(args.output)
    original=json.loads((source/'protocol.json').read_text())
    if original['sources'] != {p:digest(ROOT/p) for p in SOURCES}:
        raise ValueError('Source changed since the original measurements')
    for relative,expected_hash in original['data_hashes'].items():
        if digest(Path(original['datapath'])/relative)!=expected_hash:
            raise ValueError('Data changed: '+relative)
    for dataset,expected_hash in original['indices_hashes'].items():
        if digest(ROOT/f'data/indices109/{dataset}_INDICES_TRAIN.txt')!=expected_hash:
            raise ValueError('Resample indices changed: '+dataset)
    if not (source/'choice.json').exists():
        raise ValueError('Selection must be complete and frozen')
    choice=json.loads((source/'choice.json').read_text())
    extra=freeze_conservative(source)
    if args.freeze_only:
        print(json.dumps(extra,indent=2)); return
    expected=[source/'confirmation'/f'{d}__{v}__0.json'
              for d in original['split']['heldout'] for v in choice['finalists']]
    if not all(p.exists() for p in expected):
        raise ValueError('Original confirmation pass must finish before reuse')
    if target.exists():
        protocol=json.loads((target/'protocol.json').read_text())
        if protocol['selection_protocol_sha256'] != digest(source/'protocol.json'):
            raise ValueError('The source protocol differs from the reused measurements')
        if protocol['repeats'] != args.repeats:
            raise ValueError('Timing repetition count differs from frozen protocol')
        if protocol['repetition_driver_sha256']!=digest(__file__):
            raise ValueError('Repetition driver changed; use a new output directory')
    else:
        target.mkdir(parents=True)
        protocol={**original,'repeats':args.repeats,
                  'selection_reused_from':str(source.resolve()),
                  'repeat0_reused_from':str(source.resolve()),
                  'selection_protocol_sha256':digest(source/'protocol.json'),
                  'extra_choice_sha256':digest(source/'extra_choice.json'),
                  'repetition_driver_sha256':digest(__file__)}
        atomic_json(target/'protocol.json',protocol)
        for folder in ('selection','confirmation'):
            shutil.copytree(source/folder,target/folder)
        for file in ('choice.json','selection_summary.csv','extra_choice.json'):
            shutil.copy2(source/file,target/file)
    if digest(target/'extra_choice.json')!=digest(source/'extra_choice.json'):
        raise ValueError('Conservative choice changed during reuse')
    variants=list(dict.fromkeys(choice['finalists']+[extra['conservative']]))
    args.output=str(target)
    args.max_rss_gib=protocol['max_rss_gib']
    datasets=protocol['split']['heldout'] if args.stage=='confirmation' else ['ElectricDevices']
    repetitions=args.repeats if args.stage=='confirmation' else args.stress_repeats
    for i,dataset in enumerate(datasets):
        for repeat in range(repetitions):
            shift=(i+repeat)%len(variants)
            for variant in variants[shift:]+variants[:shift]:
                print(f'{args.stage} {i+1}/{len(datasets)} {dataset} {variant} repeat={repeat}',flush=True)
                run_job(args,args.stage,dataset,variant,repeat)


if __name__=='__main__':
    main()
