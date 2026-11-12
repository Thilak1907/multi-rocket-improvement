"""Reproducible dataset-held-out selection and isolated ablation measurements.

Run from the repository root with: python -m experiments.compare_optimizations
Each dataset/variant is a fresh subprocess; existing successful jobs resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('multirocket/multirocket.py', 'multirocket/pooling.py',
           'multirocket/fast_ridge.py', 'experiments/compare_optimizations.py')
VARIANTS = ('baseline', 'pooling_only', 'ridge_only', 'memory_only', 'pooling_ridge', 'combined')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def seed_for(dataset, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{dataset}'.encode()).digest()[:4], 'little')


def load_data(dataset, protocol):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    folder = Path(protocol['datapath']) / dataset
    train = np.loadtxt(folder / f'{dataset}_TRAIN.tsv', delimiter='\t')
    test = np.loadtxt(folder / f'{dataset}_TEST.tsv', delimiter='\t')
    encoder = LabelEncoder().fit(train[:, 0])
    # Match the repo's missing-value policy, but make it reproducible. These
    # 109 datasets have fixed lengths; the manifest records input hashes.
    rng = np.random.default_rng(seed_for(dataset, protocol['seed']))
    X, Xt = train[:, 1:].copy(), test[:, 1:].copy()
    for a in (X, Xt):
        mask = np.isnan(a)
        a[mask] = rng.random(mask.sum()) / 1000
    return X, encoder.transform(train[:, 0]), Xt, encoder.transform(test[:, 0])


def initialize(args):
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import numba
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'protocol.json'
    sources = {p: digest(ROOT / p) for p in SOURCES}
    if manifest.exists():
        protocol = json.loads(manifest.read_text())
        if protocol['sources'] != sources:
            raise ValueError('Source changed since protocol was frozen; use a new output directory.')
        for key in ('num_features', 'seed', 'split_seed', 'threads', 'repeats', 'batch_size', 'max_rss_gib'):
            if protocol[key] != getattr(args, key):
                raise ValueError(f'{key} differs from the frozen protocol')
        if protocol['datapath'] != str(Path(args.datapath).resolve()):
            raise ValueError('Data path differs from the frozen protocol')
        for path, expected in protocol['data_hashes'].items():
            if digest(Path(protocol['datapath']) / path) != expected:
                raise ValueError(f'Dataset changed: {path}')
        return protocol
    datasets = sorted(pd.read_csv(ROOT / 'data/classification_datasets_109.csv')['Name'])
    # Preserve the already established split, without reusing its partial scores.
    previous = ROOT / 'output/pooling_ablation/50k_split42/protocol.json'
    if args.split_seed == 42 and previous.exists():
        split = json.loads(previous.read_text())['split']
        if sorted(split['selection'] + split['heldout']) != datasets:
            raise ValueError('Existing split does not match the dataset universe')
    else:
        permuted = np.random.default_rng(args.split_seed).permutation(datasets)
        split = {'selection': sorted(permuted[:len(datasets)//2]),
                 'heldout': sorted(permuted[len(datasets)//2:])}
    protocol = {key: getattr(args, key) for key in
                ('num_features', 'seed', 'split_seed', 'threads', 'repeats', 'batch_size', 'max_rss_gib')}
    protocol.update(datapath=str(Path(args.datapath).resolve()), split=split,
                    selection_rule='Maximum unweighted mean accuracy on selection datasets; PAIRS order breaks ties',
                    selection_classifier='FastRidgeClassifierCV, same default LOO-MSE objective as sklearn',
                    resamples=[0], variants=VARIANTS, sources=sources,
                    python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, numba=numba.__version__,
                    timing='Warm JIT; wall clock fit and predict; subprocess startup and data load excluded',
                    memory='Process peak RSS including imports, data, and JIT warmup; 50ms supervisor guard',
                    data_hashes={f'{d}/{d}_{part}.tsv': digest(Path(args.datapath) / d / f'{d}_{part}.tsv')
                                 for d in datasets for part in ('TRAIN', 'TEST')})
    atomic_json(manifest, protocol)
    return protocol


def warmup():
    import numpy as np
    from multirocket.multirocket import MultiRocket
    for pooling in (('PPV', 'LSPV', 'MPV', 'MIPV'), ('PPV', 'LSPV')):
        model = MultiRocket(672, pooling=pooling, random_state=1, classifier='fast_ridge', memory_efficient=True)
        model.fit(np.random.default_rng(1).normal(size=(8, 32)), np.arange(8) % 2, False)
        model.predict(np.zeros((2, 32)))


def worker(args):
    import gc
    import resource
    import numpy as np
    import numba
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from multirocket.fast_ridge import FastRidgeClassifierCV
    from multirocket.multirocket import MultiRocket, fit, seed_numba, transform
    from multirocket.pooling import OPERATORS, PAIRS, feature_indices
    output = Path(args.output)
    protocol = json.loads((output / 'protocol.json').read_text())
    numba.set_num_threads(protocol['threads'])
    with threadpool_limits(limits=1, user_api='blas'):
        warmup()
        X, y, Xt, yt = load_data(args.dataset, protocol)
        seed = seed_for(args.dataset, protocol['seed'])
        if args.worker == 'selection':
            seed_numba(seed)
            p = fit(X, protocol['num_features'] // 8)
            p1 = fit(np.diff(X), protocol['num_features'] // 8)
            full = transform(X, np.diff(X), p, p1, 4)
            np.nan_to_num(full, copy=False)
            classifiers = []
            rows = []
            for pair in PAIRS:
                columns = feature_indices(p, p1, pair)
                features = np.ascontiguousarray(full[:, columns])
                model = make_pipeline(StandardScaler(copy=False), FastRidgeClassifierCV())
                t = time.perf_counter()
                model.fit(features, y)
                rows.append(dict(dataset=args.dataset, operators='+'.join(pair),
                                 features=features.shape[1], alpha=float(model[-1].alpha_),
                                 classifier_fit_s=time.perf_counter() - t, seed=seed))
                classifiers.append((model, columns))
                del features
                gc.collect()
            del full
            correct = np.zeros(len(PAIRS), dtype=np.int64)
            for start in range(0, len(Xt), protocol['batch_size']):
                chunk = Xt[start:start + protocol['batch_size']]
                full = transform(chunk, np.diff(chunk), p, p1, 4)
                np.nan_to_num(full, copy=False)
                for i, (model, columns) in enumerate(classifiers):
                    pred = model.predict(np.ascontiguousarray(full[:, columns]))
                    correct[i] += np.sum(pred == yt[start:start + len(chunk)])
                del full
            for row, count in zip(rows, correct):
                row['accuracy'] = float(count / len(yt))
            atomic_json(output / 'selection' / f'{args.dataset}.json', rows)
            return
        choice = json.loads((output / 'choice.json').read_text())['operators']
        variant = args.variant
        pooling = choice if variant in ('pooling_only', 'pooling_ridge', 'combined') else OPERATORS
        backend = 'fast_ridge' if variant in ('ridge_only', 'pooling_ridge', 'combined') else 'ridge'
        efficient = variant in ('memory_only', 'combined')
        model = MultiRocket(protocol['num_features'], classifier=backend, pooling=pooling,
                            random_state=seed, memory_efficient=efficient,
                            prediction_batch_size=protocol['batch_size'])
        t = time.perf_counter()
        model.fit(X, y, predict_on_train=False)
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        pred = model.predict(Xt)
        predict_s = time.perf_counter() - t
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
        row = dict(dataset=args.dataset, variant=variant, repeat=args.repeat, status='ok',
                   seed=seed, accuracy=float(np.mean(pred == yt)), alpha=float(model.classifier[-1].alpha_),
                   features=model.n_features_out_, train_feature_bytes=model.train_feature_bytes_,
                   max_prediction_feature_bytes=model.max_prediction_feature_bytes_,
                   fit_s=fit_s, predict_s=predict_s, total_s=fit_s + predict_s,
                   classifier_fit_s=model.train_duration, classifier_predict_s=model.test_duration,
                   train_transform_s=model.apply_kernel_on_train_duration,
                   test_transform_s=model.apply_kernel_on_test_duration, peak_rss_bytes=rss_bytes,
                   prediction_hash=hashlib.sha256(pred.tobytes()).hexdigest())
        target = output / 'evaluation' / f'{args.dataset}__{variant}__{args.repeat}'
        np.save(target.with_suffix('.npy'), pred)
        atomic_json(target.with_suffix('.json'), row)


def run_job(args, stage, dataset, variant='', repeat=0):
    import psutil
    output = Path(args.output)
    folder = output / stage
    folder.mkdir(exist_ok=True)
    stem = dataset if stage == 'selection' else f'{dataset}__{variant}__{repeat}'
    target = folder / f'{stem}.json'
    if target.exists():
        value = json.loads(target.read_text())
        if isinstance(value, list) or value.get('status') == 'ok':
            return True
        # A recorded resource guard is an observed failure, not an invitation
        # to retry the same unsafe allocation during resume.
        if value.get('status') == 'rss_limit':
            return False
    cmd = [sys.executable, '-m', 'experiments.compare_optimizations', '--worker', stage,
           '--output', str(output), '--dataset', dataset, '--variant', variant, '--repeat', str(repeat)]
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (folder / f'{stem}.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        peak = 0
        killed = False
        while child.poll() is None:
            try:
                peak = max(peak, process.memory_info().rss)
                if peak > args.max_rss_gib * 1024**3:
                    child.kill()
                    killed = True
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
        if child.returncode != 0:
            atomic_json(target, dict(dataset=dataset, variant=variant, repeat=repeat,
                                     status='rss_limit' if killed else 'failed', exitcode=child.returncode,
                                     peak_rss_bytes=peak))
            print(f'  FAILED {stem}: ' + ('RSS guard' if killed else f'exit {child.returncode}'), flush=True)
            return False
    return True


def freeze_choice(output, protocol):
    import pandas as pd
    from multirocket.pooling import PAIRS
    rows = []
    for dataset in protocol['split']['selection']:
        value = json.loads((output / 'selection' / f'{dataset}.json').read_text())
        if not isinstance(value, list) or len(value) != 6:
            raise RuntimeError('All 54 selection datasets must finish before choosing a pair')
        if {r['operators'] for r in value} != {'+'.join(p) for p in PAIRS}:
            raise RuntimeError('Incomplete candidate grid')
        rows.extend(value)
    df = pd.DataFrame(rows)
    means = df.groupby('operators')['accuracy'].mean()
    pair = max(PAIRS, key=lambda p: means['+'.join(p)])
    choice = dict(operators=list(pair), selection_mean_accuracy=float(means['+'.join(pair)]),
                  candidate_means=means.to_dict(), selection_datasets=len(protocol['split']['selection']),
                  selection_files_sha256={d: digest(output / 'selection' / f'{d}.json')
                                          for d in protocol['split']['selection']})
    target = output / 'choice.json'
    if target.exists() and json.loads(target.read_text()) != choice:
        raise RuntimeError('Refusing to change an already frozen pair')
    atomic_json(target, choice)
    df.to_csv(output / 'selection_scores.csv', index=False)
    return choice


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='output/optimization_comparison/50k_split42')
    p.add_argument('--datapath', default='data/UCRArchive_2018')
    p.add_argument('--num-features', type=int, default=50000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-rss-gib', type=float, default=9.0)
    p.add_argument('--stage', choices=('all', 'selection', 'evaluation', 'stress'), default='all')
    p.add_argument('--worker', choices=('selection', 'evaluation'))
    p.add_argument('--dataset')
    p.add_argument('--variant', default='')
    p.add_argument('--repeat', type=int, default=0)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    protocol = initialize(args)
    output = Path(args.output)
    if args.stage in ('all', 'selection'):
        # Size ordering is predeclared, based on training rows and series length,
        # never on accuracy. It only affects job order.
        import pandas as pd
        sizes = pd.read_csv(ROOT / 'data/classification_datasets_109.csv').set_index('Name')
        datasets = sorted(protocol['split']['selection'], key=lambda d: sizes.loc[d, 'Train '] * sizes.loc[d, 'Length'])
        for i, dataset in enumerate(datasets):
            print(f'selection {i+1}/{len(datasets)} {dataset}', flush=True)
            if not run_job(args, 'selection', dataset):
                raise RuntimeError('Selection incomplete; held-out evaluation remains locked')
        choice = freeze_choice(output, protocol)
        print('Frozen pair: ' + '+'.join(choice['operators']), flush=True)
    if args.stage in ('all', 'evaluation', 'stress'):
        # Recompute only from selection records to verify the lock before
        # permitting any held-out labels into the pipeline.
        freeze_choice(output, protocol)
        datasets = ['ElectricDevices'] if args.stage == 'stress' else protocol['split']['heldout']
        if args.stage == 'all':
            datasets = datasets + ['ElectricDevices']  # separately reported stress case, not held-out
        for i, dataset in enumerate(datasets):
            for repeat in range(protocol['repeats']):
                # Rotate execution order to reduce temperature/order bias.
                shift = (i + repeat) % len(VARIANTS)
                order = VARIANTS[shift:] + VARIANTS[:shift]
                for variant in order:
                    print(f'evaluation {i+1}/{len(datasets)} {dataset} {variant} repeat={repeat}', flush=True)
                    run_job(args, 'evaluation', dataset, variant, repeat)


if __name__ == '__main__':
    main()
