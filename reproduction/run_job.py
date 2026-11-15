"""Run one (configuration, dataset) job over a list of (resample, seed) pairs.

Runs in its own process so that peak RSS belongs to this job only. Prints one
JSON line per run, then one line with the process peak RSS.

Usage:
    python reproduction/run_job.py CONFIG DATASET PAIRS [--data-dir DIR] [--indices-dir DIR]

PAIRS is a comma-separated list of resample:seed, e.g. "0:0,0:1" or "1:0,2:0".
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from multirocket.multirocket import MultiRocket  # noqa: E402
from utils.data_loader import read_univariate_ucr  # noqa: E402

CONFIGS = {
    # Published configuration: 50k, scikit-learn RidgeClassifierCV, original first difference
    "paper50": dict(num_features=50000, classifier="ridge", first_difference_fix=False),
    # Same, with the corrected first-difference alignment
    "fixed50": dict(num_features=50000, classifier="ridge", first_difference_fix=True),
    # Retained configuration (same settings as the README's recommended command)
    "rec30": dict(num_features=30000, classifier="fast_ridge", first_difference_fix=True,
                  memory_efficient=True, prediction_batch_size=256,
                  optimized_transform=True, fused_prediction=True),
    # Speed-first alternative
    "fast10": dict(num_features=10000, classifier="fast_ridge", first_difference_fix=True,
                   memory_efficient=True, prediction_batch_size=256,
                   optimized_transform=True, fused_prediction=True),
}


def load(data_dir, indices_dir, problem, resample):
    """Load a dataset; resample 0 is the original split, others use the authors' index files."""
    xtr, ytr = read_univariate_ucr(f"{data_dir}/{problem}/{problem}_TRAIN.tsv", normalise=False)
    xte, yte = read_univariate_ucr(f"{data_dir}/{problem}/{problem}_TEST.tsv", normalise=False)
    enc = LabelEncoder()
    ytr = enc.fit_transform(ytr)
    yte = enc.transform(yte)
    xtr = xtr.reshape(xtr.shape[0], -1)
    xte = xte.reshape(xte.shape[0], -1)
    if resample > 0:  # same logic as main_ucr.py
        x = np.vstack((xtr, xte))
        y = np.hstack((ytr, yte))
        train = np.loadtxt(f"{indices_dir}/{problem}_INDICES_TRAIN.txt",
                           skiprows=resample, max_rows=1).astype(np.int32)
        test = np.setdiff1d(np.arange(len(x)), train, assume_unique=True)
        xtr, ytr, xte, yte = x[train], y[train], x[test], y[test]
    return xtr, ytr, xte, yte


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", choices=sorted(CONFIGS))
    parser.add_argument("dataset")
    parser.add_argument("pairs")
    parser.add_argument("--data-dir", default=str(ROOT / "data/UCRArchive_2018"))
    parser.add_argument("--indices-dir", default=str(ROOT / "data/indices109"))
    args = parser.parse_args()
    pairs = [tuple(map(int, p.split(":"))) for p in args.pairs.split(",")]

    # Numba warm-up on a tiny problem, excluded from timings
    rng = np.random.default_rng(0)
    warm = MultiRocket(**{**CONFIGS[args.config], "num_features": 1000}, random_state=0)
    warm.fit(rng.standard_normal((10, 40)), np.arange(10) % 2, predict_on_train=False)
    warm.predict(rng.standard_normal((5, 40)))

    for resample, seed in pairs:
        xtr, ytr, xte, yte = load(args.data_dir, args.indices_dir, args.dataset, resample)
        model = MultiRocket(**CONFIGS[args.config], random_state=seed)
        t0 = time.perf_counter()
        model.fit(xtr, ytr, predict_on_train=False)
        t1 = time.perf_counter()
        yhat = model.predict(xte)
        t2 = time.perf_counter()
        print(json.dumps(dict(
            config=args.config, dataset=args.dataset, resample=resample, seed=seed,
            acc=float(accuracy_score(yte, yhat)), fit_s=t1 - t0, predict_s=t2 - t1,
            transform_train_s=float(getattr(model, "apply_kernel_on_train_duration", np.nan)),
            classifier_fit_s=float(getattr(model, "train_duration", np.nan)),
        )), flush=True)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KiB
    if sys.platform == "darwin":  # macOS reports bytes
        peak_mib /= 1024
    print(json.dumps(dict(config=args.config, dataset=args.dataset, peak_rss_mib=peak_mib)), flush=True)


if __name__ == "__main__":
    main()
