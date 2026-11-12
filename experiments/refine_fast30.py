"""Check the 30k conservative profile with blocked ridge after a memory warning.

This refinement is motivated by selection-only ElectricDevices measurements:
fused30 reached 8.91 GiB, whereas fast50 used 6.47 GiB. All 54 selection
predictions and alphas must match fused30 before confirmation is unlocked.
The initial search, choice, and repetition protocols are left immutable.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.compare_optimizations import atomic_json, digest
from experiments import search_combinations as search

CONFIG={**search.CONFIGS['fused30'],'classifier':'fast_ridge'}


def initialize(args):
    source,output=Path(args.source),Path(args.output)
    original=json.loads((source/'protocol.json').read_text())
    if original['sources']!={p:digest(search.ROOT/p) for p in search.SOURCES}:
        raise ValueError('Core source differs from the selection experiment')
    if args.max_rss_gib!=original['max_rss_gib']:
        raise ValueError('Memory guard differs from the original experiment')
    for relative,expected_hash in original['data_hashes'].items():
        if digest(Path(original['datapath'])/relative)!=expected_hash:
            raise ValueError('Data changed: '+relative)
    for dataset,expected_hash in original['indices_hashes'].items():
        if digest(search.ROOT/f'data/indices109/{dataset}_INDICES_TRAIN.txt')!=expected_hash:
            raise ValueError('Resample indices changed: '+dataset)
    expected={**original,'configs':{**original['configs'],'fast30':CONFIG},
              'repeats':3,'refinement_driver_sha256':digest(__file__),
              'refinement_source':str(source.resolve()),
              'refinement_parent_sha256':digest(source/'protocol.json'),
              'refinement_reason':'Selection ElectricDevices fused30 peak 8.91 GiB; fast50 6.47 GiB; verify the same 30k representation with blocked ridge',
              'refinement_rule':'Require all 54 selection jobs to succeed with identical alpha and predictions to fused30; then confirm on all 55 resample-1 datasets'}
    # Normalize tuples to their on-disk JSON representation.
    expected=json.loads(json.dumps(expected))
    output.mkdir(parents=True,exist_ok=True)
    file=output/'protocol.json'
    if file.exists() and json.loads(file.read_text())!=expected:
        raise ValueError('Refinement protocol changed; use a new output directory')
    if not file.exists(): atomic_json(file,expected)
    return expected


def job(args,phase,dataset,repeat):
    import psutil
    output=Path(args.output)
    folder=output/phase
    folder.mkdir(exist_ok=True)
    stem=f'{dataset}__fast30__{repeat}'
    target=folder/f'{stem}.json'
    if target.exists(): return json.loads(target.read_text())['status']=='ok'
    cmd=[sys.executable,'-m','experiments.refine_fast30','--worker',phase,
         '--output',str(output),'--dataset',dataset,'--repeat',str(repeat)]
    env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    with (folder/f'{stem}.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=search.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid)
        peak,status,start=0,'failed',time.monotonic()
        while child.poll() is None:
            try:
                peak=max(peak,process.memory_info().rss)
                if peak>args.max_rss_gib*1024**3:
                    status='rss_limit'; child.kill()
                elif time.monotonic()-start>1200:
                    status='timeout'; child.kill()
            except psutil.NoSuchProcess:
                pass
            time.sleep(.05)
        if child.returncode!=0:
            atomic_json(target,dict(dataset=dataset,variant='fast30',repeat=repeat,status=status,
                                    peak_rss_bytes=peak,exitcode=child.returncode))
            return False
    return target.exists()


def freeze(output,source,protocol):
    hashes={}
    for dataset in protocol['split']['selection']:
        file=output/'selection'/f'{dataset}__fast30__0.json'
        actual=json.loads(file.read_text())
        reference=json.loads((source/'selection'/f'{dataset}__fused30__0.json').read_text())
        if actual['status']!='ok' or reference['status']!='ok':
            raise ValueError('Incomplete refinement selection: '+dataset)
        for key in ('alpha','prediction_hash','train_indices_sha256','test_indices_sha256'):
            if actual[key]!=reference[key]:
                raise ValueError('Refinement parity mismatch '+dataset+': '+key)
        hashes[dataset]=dict(actual=digest(file),reference=digest(source/'selection'/f'{dataset}__fused30__0.json'))
    value=dict(variant='fast30',reference='fused30',selection_jobs=54,
               alpha_matches=54,prediction_matches=54,selection_hashes=hashes)
    file=output/'choice.json'
    if file.exists() and json.loads(file.read_text())!=value:
        raise ValueError('Refinement choice differs from its frozen inputs')
    if not file.exists(): atomic_json(file,value)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='output/combination_search/20261111')
    parser.add_argument('--output',default='output/combination_search/20261111_fast30')
    parser.add_argument('--stage',choices=('all','selection','confirmation','stress'),default='all')
    parser.add_argument('--worker',choices=('selection','confirmation','stress'))
    parser.add_argument('--dataset')
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--max-rss-gib',type=float,default=9)
    args=parser.parse_args()
    if args.worker:
        search.CONFIGS['fast30']=CONFIG
        args.variant='fast30'
        search.worker(args); return
    protocol=initialize(args)
    source,output=Path(args.source),Path(args.output)
    if args.stage in ('all','selection'):
        for i,dataset in enumerate(protocol['split']['selection']):
            print(f'selection {i+1}/54 {dataset} fast30',flush=True)
            job(args,'selection',dataset,0)
        freeze(output,source,protocol)
    if args.stage in ('all','confirmation','stress'):
        freeze(output,source,protocol)
        phase='stress' if args.stage=='stress' else 'confirmation'
        datasets=['ElectricDevices'] if phase=='stress' else protocol['split']['heldout']
        for i,dataset in enumerate(datasets):
            for repeat in range(1 if phase=='stress' else protocol['repeats']):
                print(f'{phase} {i+1}/{len(datasets)} {dataset} fast30 repeat={repeat}',flush=True)
                job(args,phase,dataset,repeat)


if __name__=='__main__':
    main()
