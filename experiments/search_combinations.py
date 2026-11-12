"""Frozen selection of accuracy/time/memory tradeoffs, followed by resample 1.

Selection: class-stratified 80/20 validation inside the official TRAIN files of
the original 54 selection datasets. No official TEST labels enter selection.
Confirmation: the original 55 held-out dataset names, official resample 1.
Every measurement is an isolated CPU process, with warm JIT and a 9 GiB guard.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

from experiments.compare_optimizations import atomic_json, digest, seed_for, load_data

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(num_features=50000, classifier='ridge', random_state=None,
            memory_efficient=True, prediction_batch_size=256,
            optimized_transform=True, fused_prediction=True)
CONFIGS = {
    'baseline50': dict(num_features=50000, classifier='ridge', memory_efficient=False),
    'buffered50': {**BASE, 'fused_prediction': False},
    'fused50': dict(BASE),
    'fused30': {**BASE, 'num_features': 30000},
    'fused20': {**BASE, 'num_features': 20000},
    'fused10': {**BASE, 'num_features': 10000},
    'fast50': {**BASE, 'classifier': 'fast_ridge'},
    'pair50': dict(num_features=50000, classifier='fast_ridge', pooling=('LSPV', 'MPV'),
                   memory_efficient=True, prediction_batch_size=256, fused_prediction=True),
    'hydra64': dict(hydra_groups=64, num_features=0),
    'hybrid20h32': dict(hydra_groups=32, num_features=20000),
}
SOURCES = ['multirocket/multirocket.py', 'multirocket/fast_ridge.py',
           'multirocket/pooling.py', 'multirocket/buffered.py',
           'experiments/search_combinations.py', 'experiments/hydra_model.py',
           'experiments/vendor/hydra.py']


def initialize(args):
    import numpy as np
    import scipy, sklearn, numba, torch
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sources = {f: digest(ROOT / f) for f in SOURCES}
    previous = json.loads((ROOT / 'output/optimization_comparison/50k_split42_alpha_tie/protocol.json').read_text())
    manifest = output / 'protocol.json'
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if value['sources'] != sources or value['configs'] != json.loads(json.dumps(CONFIGS)):
            raise ValueError('Implementation changed; use a new output directory')
        if value['repeats'] != args.repeats or value['max_rss_gib'] != args.max_rss_gib:
            raise ValueError('Requested settings differ from frozen protocol')
        return value
    value = dict(sources=sources, configs=CONFIGS, split=previous['split'],
                 seed=20261111, threads=4, blas_threads=1, batch_size=256,
                 repeats=args.repeats, max_rss_gib=args.max_rss_gib,
                 datapath=previous['datapath'], data_hashes=previous['data_hashes'],
                 indices_hashes={d: digest(ROOT / f'data/indices109/{d}_INDICES_TRAIN.txt')
                                  for d in previous['split']['heldout']},
                 selection='20% per class from official TRAIN only; at least one training item per class',
                 confirmation='official resample 1 for the original 55 held-out dataset names',
                 accuracy_tolerance_pp=0.5,
                 balanced_rule='Within 0.5 percentage points of baseline mean validation accuracy; minimize sqrt(total time ratio * median paired peak RSS ratio). All selection jobs must succeed.',
                 finalist_rule='baseline, balanced, accuracy winner, fastest eligible, full-budget fused50; deduplicate',
                 timing='Warm CPU fit+predict; data loading, imports, startup, serialization excluded',
                 memory='Process peak RSS includes imports and warmup; supervisor polls at 50ms',
                 platform=platform.platform(), python=sys.version,
                 versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                               numba=numba.__version__, torch=torch.__version__))
    for relative, expected in value['data_hashes'].items():
        if digest(Path(value['datapath']) / relative) != expected:
            raise ValueError('Data changed: '+relative)
    atomic_json(manifest, value)
    return value


def data_for(dataset, phase, protocol):
    import numpy as np
    X, y, Xt, yt = load_data(dataset, protocol)
    rng = np.random.default_rng(seed_for(dataset, protocol['seed']))
    if phase == 'selection':
        keep, val = [], []
        for label in np.unique(y):
            indices = rng.permutation(np.flatnonzero(y == label))
            nval = min(len(indices)-1, max(1, int(round(.2*len(indices)))))
            val.extend(indices[:nval])
            keep.extend(indices[nval:])
        keep, val = np.sort(keep), np.sort(val)
        return X[keep], y[keep], X[val], y[val], dict(
            train_indices_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
            test_indices_sha256=hashlib.sha256(val.tobytes()).hexdigest())
    all_X, all_y = np.concatenate((X, Xt)), np.concatenate((y, yt))
    train = np.loadtxt(ROOT / f'data/indices109/{dataset}_INDICES_TRAIN.txt',
                       skiprows=1, max_rows=1).astype(np.int64)
    if len(np.unique(train)) != len(X) or train.min() < 0 or train.max() >= len(all_X):
        raise ValueError('Invalid resample indices for '+dataset)
    test = np.setdiff1d(np.arange(len(all_X)), train)
    return all_X[train], all_y[train], all_X[test], all_y[test], dict(
        train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
        test_indices_sha256=hashlib.sha256(test.tobytes()).hexdigest())


def warmup():
    import numpy as np
    from multirocket.multirocket import MultiRocket
    from experiments.hydra_model import HydraRocket
    X = np.random.default_rng(0).normal(size=(8,32))
    for config in (CONFIGS['baseline50'], CONFIGS['fused50'], CONFIGS['pair50']):
        options = {**config, 'num_features': 672, 'random_state': 1}
        model = MultiRocket(**options)
        model.fit(X, np.arange(8) % 2, False)
        model.predict(X[:2])
    model = HydraRocket(groups=4, seed=1)
    model.fit(X, np.arange(8) % 2)
    model.predict(X[:2])


def worker(args):
    import numpy as np
    import numba, torch, resource
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from threadpoolctl import threadpool_limits
    from multirocket.multirocket import MultiRocket
    from experiments.hydra_model import HydraRocket
    output = Path(args.output)
    protocol = json.loads((output/'protocol.json').read_text())
    numba.set_num_threads(protocol['threads'])
    torch.set_num_threads(protocol['threads'])
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=protocol['blas_threads'], user_api='blas'):
        warmup()
        X,y,Xt,yt,split_hashes = data_for(args.dataset, args.worker, protocol)
        config = dict(CONFIGS[args.variant])
        seed = seed_for(args.dataset, protocol['seed'])
        if 'hydra_groups' in config:
            model = HydraRocket(budget=config['num_features'], groups=config['hydra_groups'], seed=seed)
        else:
            config['random_state'] = seed
            model = MultiRocket(**config)
        t = time.perf_counter()
        model.fit(X,y,predict_on_train=False)
        fit_s = time.perf_counter()-t
        t = time.perf_counter()
        predictions = model.predict(Xt)
        predict_s = time.perf_counter()-t
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == 'darwin' else 1024
        alpha = model.alpha_ if isinstance(model, HydraRocket) else float(model.classifier[-1].alpha_)
        row = dict(dataset=args.dataset, variant=args.variant, repeat=args.repeat, status='ok',
                   n_train=len(X), n_test=len(Xt), length=X.shape[1], seed=seed, **split_hashes,
                   accuracy=float(accuracy_score(yt,predictions)),
                   balanced_accuracy=float(balanced_accuracy_score(yt,predictions)),
                   macro_f1=float(f1_score(yt,predictions,average='macro',zero_division=0)),
                   fit_s=fit_s, predict_s=predict_s, total_s=fit_s+predict_s,
                   classifier_fit_s=model.train_duration, classifier_predict_s=model.test_duration,
                   train_transform_s=model.apply_kernel_on_train_duration,
                   test_transform_s=model.apply_kernel_on_test_duration,
                   peak_rss_bytes=rss, features=model.n_features_out_, alpha=alpha,
                   train_feature_bytes=model.train_feature_bytes_,
                   max_prediction_feature_bytes=model.max_prediction_feature_bytes_,
                   prediction_hash=hashlib.sha256(predictions.tobytes()).hexdigest(),
                   model_bytes=len(pickle.dumps(model,protocol=5)))
        target = output/args.worker/f'{args.dataset}__{args.variant}__{args.repeat}'
        np.save(target.with_suffix('.npy'),predictions)
        atomic_json(target.with_suffix('.json'),row)


def run_job(args, phase, dataset, variant, repeat):
    import psutil
    output = Path(args.output)
    folder = output/phase
    folder.mkdir(exist_ok=True)
    stem = f'{dataset}__{variant}__{repeat}'
    target = folder/f'{stem}.json'
    if target.exists():
        return json.loads(target.read_text())['status'] == 'ok'
    cmd = [sys.executable,'-m','experiments.search_combinations','--worker',phase,
           '--output',str(output),'--dataset',dataset,'--variant',variant,'--repeat',str(repeat)]
    env = dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    with (folder/f'{stem}.log').open('w') as log:
        child = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        peak, status = 0, 'failed'
        start=time.monotonic()
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
        if child.returncode != 0:
            atomic_json(target,dict(dataset=dataset,variant=variant,repeat=repeat,
                                    status=status,peak_rss_bytes=peak,exitcode=child.returncode))
            print(f'  {status}: {stem}',flush=True)
            return False
    if not target.exists():
        raise RuntimeError('Worker completed without a result: '+stem)
    return True


def selection_summary(output, protocol):
    import numpy as np
    import pandas as pd
    expected = [(d,v) for d in protocol['split']['selection'] for v in CONFIGS]
    paths = [output/'selection'/f'{d}__{v}__0.json' for d,v in expected]
    raw = pd.DataFrame([json.loads(p.read_text()) for p in paths])
    baseline=raw[(raw.variant=='baseline50') & (raw.status=='ok')].set_index('dataset')
    rows=[]
    for variant in CONFIGS:
        current=raw[(raw.variant==variant)&(raw.status=='ok')].set_index('dataset')
        names=baseline.index.intersection(current.index)
        b,v=baseline.loc[names],current.loc[names]
        if not len(names):
            continue
        time_ratio=float(v.total_s.sum()/b.total_s.sum())
        rss_ratio=float((v.peak_rss_bytes/b.peak_rss_bytes).median())
        rows.append(dict(variant=variant,datasets=len(names),accuracy=float(v.accuracy.mean()),
                         delta_pp=float((v.accuracy-b.accuracy).mean()*100),
                         time_ratio=time_ratio,rss_ratio=rss_ratio,
                         cost_score=float(np.sqrt(time_ratio*rss_ratio)),
                         successful=int((raw[(raw.variant==variant)].status=='ok').sum())))
    summary=pd.DataFrame(rows)
    eligible=summary[(summary.delta_pp>=-protocol['accuracy_tolerance_pp'])&
                     (summary.datasets==len(protocol['split']['selection'])) &
                     (summary.successful==len(protocol['split']['selection']))]
    if not len(eligible):
        raise RuntimeError('No complete candidate meets the declared accuracy tolerance')
    balanced=eligible.sort_values(['cost_score','variant']).iloc[0].variant
    accurate=summary[summary.datasets==len(protocol['split']['selection'])].sort_values(
        ['accuracy','time_ratio','variant'],ascending=[False,True,True]).iloc[0].variant
    fastest=eligible.sort_values(['time_ratio','variant']).iloc[0].variant
    finalists=list(dict.fromkeys(['baseline50',balanced,accurate,fastest,'fused50']))
    choice=dict(balanced=balanced,accuracy_winner=accurate,fastest_eligible=fastest,
                finalists=finalists,selection_files={str(p.relative_to(output)):digest(p) for p in paths},
                rule=protocol['balanced_rule'])
    target=output/'choice.json'
    if target.exists() and json.loads(target.read_text()) != choice:
        raise ValueError('Frozen choice differs from recomputed selection')
    atomic_json(target,choice)
    summary.to_csv(output/'selection_summary.csv',index=False)
    return choice


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/combination_search/20261111')
    parser.add_argument('--stage',choices=('all','selection','confirmation','stress'),default='all')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--max-rss-gib',type=float,default=9)
    parser.add_argument('--worker',choices=('selection','confirmation','stress'))
    parser.add_argument('--dataset')
    parser.add_argument('--variant',choices=tuple(CONFIGS))
    parser.add_argument('--repeat',type=int,default=0)
    args=parser.parse_args()
    if args.worker:
        worker(args); return
    protocol=initialize(args)
    output=Path(args.output)
    if args.stage in ('all','selection'):
        for i,dataset in enumerate(protocol['split']['selection']):
            variants=list(CONFIGS)
            shift=i%len(variants)
            for variant in variants[shift:]+variants[:shift]:
                print(f'selection {i+1}/54 {dataset} {variant}',flush=True)
                run_job(args,'selection',dataset,variant,0)
        choice=selection_summary(output,protocol)
        print('Frozen selection: '+json.dumps({k:v for k,v in choice.items() if k!='selection_files'}),flush=True)
    if args.stage in ('all','confirmation','stress'):
        choice=selection_summary(output,protocol)
        datasets=protocol['split']['heldout'] if args.stage!='stress' else ['ElectricDevices']
        for i,dataset in enumerate(datasets):
            for repeat in range(protocol['repeats']):
                variants=choice['finalists']
                shift=(i+repeat)%len(variants)
                for variant in variants[shift:]+variants[:shift]:
                    phase='stress' if args.stage=='stress' else 'confirmation'
                    print(f'{phase} {i+1}/{len(datasets)} {dataset} {variant} repeat={repeat}',flush=True)
                    run_job(args,phase,dataset,variant,repeat)


if __name__=='__main__':
    main()
