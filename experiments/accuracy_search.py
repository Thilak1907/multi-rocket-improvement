"""Train-only candidate selection, then paired accuracy confirmation.

One dataset per isolated process; common baseline reused across candidates.
RSS is a process high-water upper bound shared across candidates, not a precise
isolated per-candidate memory comparison. No changes to the production default.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from experiments.compare_optimizations import atomic_json, seed_for, digest

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ['seed', 'normalized', 'coarse']
SOURCES = ['multirocket/accuracy_ensemble.py', 'multirocket/presets.py',
           'multirocket/multirocket.py', 'multirocket/corrected.py',
           'multirocket/buffered.py', 'multirocket/fast_ridge.py',
           'multirocket/pooling.py', 'experiments/accuracy_search.py',
           'experiments/review_revision.py', 'experiments/compare_optimizations.py']


def initialize(out):
    original = json.loads((ROOT / 'output/review_revision/corrected_v1/protocol.json').read_text())
    sources = {p: digest(ROOT / p) for p in SOURCES}
    path = out / 'protocol.json'
    if path.exists():
        protocol = json.loads(path.read_text())
        if protocol['sources'] != sources:
            raise ValueError('Source changed; use a new output directory')
        return protocol
    protocol = {**original, 'sources': sources, 'ablation_train_cap': 512, 'ablation_test_cap': 512,
                'views': VIEWS, 'selection': '54 selection names, stratified TRAIN-only 80/20 validation; caps 512 per partition',
                'rule': 'Choose largest mean validation accuracy gain; VIEWS order breaks ties. Require >=0.1 pp and summed runtime <=2.5x baseline; otherwise no adoption.',
                'confirmation': '55 previously inspected confirmation names, archive resamples 1,2,3; paired per-dataset mean difference; 20000 dataset bootstrap draws, seed421',
                'adoption': 'Recommend only if confirmation mean gain >=0.1 pp and paired dataset-bootstrap 95% lower >0; runtime <=2.5x baseline; all jobs successful. Otherwise retain previous30.',
                'memory': '9 GiB process polling guard. Shared process high-water RSS, upper bound only.',
                'timing': 'warm JIT, one timing run per split; baseline reused; no timing significance claim'}
    atomic_json(path, protocol)
    for name in SOURCES:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return protocol


def worker(args):
    import numpy as np
    import numba
    import resource
    from threadpoolctl import threadpool_limits
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from experiments.review_revision import dataset
    from multirocket.presets import make_multirocket
    from multirocket.accuracy_ensemble import series_view, ridge_scores, labels_from_scores
    out = Path(args.output)
    p = json.loads((out / 'protocol.json').read_text())
    numba.set_num_threads(4)
    with threadpool_limits(limits=1):
        warm = make_multirocket(random_state=1)
        warm.fit(np.random.default_rng(1).normal(size=(8, 32)), np.arange(8) % 2, predict_on_train=False)
        ridge_scores(warm, np.ones((2, 32)))
        del warm
        phase = 'ablation' if args.worker == 'selection' else 'confirmation'
        X, y, Xt, yt, hashes = dataset(args.dataset, phase, args.resample, p)
        seed = seed_for(args.dataset, p['seed'])
        base = make_multirocket(random_state=seed)
        t = time.perf_counter(); base.fit(X, y, predict_on_train=False); base_fit = time.perf_counter() - t
        t = time.perf_counter(); base_scores = ridge_scores(base, Xt); base_predict = time.perf_counter() - t
        classes = base.classifier[-1].classes_
        base_size = len(pickle.dumps(base, protocol=5))
        def row(view, scores, fit_s, predict_s, model_bytes, alphas):
            pred = labels_from_scores(scores, classes)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            return dict(dataset=args.dataset, variant=view, resample=args.resample, status='ok',
                        **hashes, n_train=len(X), n_test=len(Xt), accuracy=float(accuracy_score(yt, pred)),
                        balanced_accuracy=float(balanced_accuracy_score(yt, pred)),
                        macro_f1=float(f1_score(yt, pred, average='macro', zero_division=0)),
                        fit_s=fit_s, predict_s=predict_s, total_s=fit_s + predict_s,
                        model_bytes=model_bytes, process_highwater_bytes=rss, alphas=alphas,
                        prediction_hash=hashlib.sha256(pred.tobytes()).hexdigest())
        rows = [row('baseline', base_scores, base_fit, base_predict, base_size, [float(base.classifier[-1].alpha_)])]
        if args.worker == 'selection':
            # Rotate ordering by dataset to reduce systematic timing-order effects.
            shift = seed % 3
            views = VIEWS[shift:] + VIEWS[:shift]
        else:
            selected = json.loads((out / 'choice.json').read_text())['selected']
            views = [selected] if selected else []
        for view in views:
            extra = make_multirocket(random_state=(seed + 104729) % (2 ** 32))
            t = time.perf_counter(); extra.fit(series_view(X, view), y, predict_on_train=False); extra_fit = time.perf_counter() - t
            t = time.perf_counter(); scores = ridge_scores(extra, series_view(Xt, view)); extra_predict = time.perf_counter() - t
            rows.append(row(view, (base_scores + scores) / 2, base_fit + extra_fit, base_predict + extra_predict,
                            base_size + len(pickle.dumps(extra, protocol=5)),
                            [float(base.classifier[-1].alpha_), float(extra.classifier[-1].alpha_)]))
            del extra, scores
            gc.collect()
        atomic_json(out / args.worker / f'{args.dataset}__s{args.resample}.json', rows)


def job(args, phase, name, split):
    import psutil
    out = Path(args.output) / phase
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{name}__s{split}.json'
    if target.exists():
        return
    env = {**os.environ, 'VECLIB_MAXIMUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '4'}
    cmd = [sys.executable, '-m', 'experiments.accuracy_search', '--output', args.output,
           '--worker', phase, '--dataset', name, '--resample', str(split)]
    print(phase, name, split, flush=True)
    with target.with_suffix('.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        proc = psutil.Process(child.pid)
        peak = 0; started = time.monotonic(); status = 'failed'
        while child.poll() is None:
            try:
                peak = max(peak, proc.memory_info().rss)
                if peak > 9 * 1024 ** 3:
                    status = 'rss_limit'; child.kill()
                elif time.monotonic() - started > 1200:
                    status = 'timeout'; child.kill()
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
        if child.returncode:
            atomic_json(target, [dict(dataset=name, resample=split, variant='all', status=status, process_highwater_bytes=peak)])
            print(status, name, flush=True)


def collect(out, phase):
    import pandas as pd
    rows = [r for path in sorted((out / phase).glob('*.json')) for r in json.loads(path.read_text())]
    return pd.DataFrame(rows)


def choose(out, protocol):
    frame = collect(out, 'selection')
    if len(frame) != 54 * 4 or not (frame.status == 'ok').all():
        raise ValueError('Selection incomplete or failed; do not select from a subset')
    pivot = frame.pivot(index='dataset', columns='variant', values='accuracy')
    times = frame.groupby('variant').total_s.sum()
    gains = {v: float((pivot[v] - pivot.baseline).mean() * 100) for v in VIEWS}
    eligible = [v for v in VIEWS if gains[v] >= .1 and times[v] / times.baseline <= 2.5]
    selected = max(eligible, key=lambda v: gains[v]) if eligible else None
    choice = dict(selected=selected, gains_pp=gains, runtime_ratios={v: float(times[v] / times.baseline) for v in VIEWS},
                  selection_results_sha256={p.name: digest(p) for p in sorted((out / 'selection').glob('*.json'))})
    file = out / 'choice.json'
    if file.exists() and json.loads(file.read_text()) != choice:
        raise ValueError('Frozen selection changed')
    atomic_json(file, choice)
    print(json.dumps(choice, indent=2)[:650], flush=True)
    return choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/accuracy_search/views_v1')
    parser.add_argument('--worker', choices=['selection', 'confirmation'])
    parser.add_argument('--dataset')
    parser.add_argument('--resample', type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        worker(args); return
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    protocol = initialize(out)
    for name in protocol['split']['selection']:
        job(args, 'selection', name, 1)
    choice = choose(out, protocol)
    if choice['selected']:
        for split in (1, 2, 3):
            for name in protocol['split']['heldout']:
                job(args, 'confirmation', name, split)
    print('FINISHED', flush=True)


if __name__ == '__main__':
    main()
