"""Run the Section IV reproduction checks and append results to a JSONL file.

Stages:
  A  original split, 10 kernel seeds, 8 small datasets, 4 configurations
  B  resamples 0-29, seed 0, the same 8 datasets and 4 configurations
  C  StarLightCurves and Crop, original split, seed 0, 4 configurations

Each job runs in its own process. A job that is killed by the operating system
(for example by the out-of-memory killer) is recorded with its exit code.

Usage:
    python reproduction/run_all.py A B C [--out reproduction/output/results.jsonl]
    python reproduction/run_all.py B --datasets Coffee,Beef --configs paper50,rec30
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMALL = ["ArrowHead", "Beef", "Coffee", "ECG200", "GunPoint", "ItalyPowerDemand",
         "OSULeaf", "InsectWingbeatSound"]
LARGE = ["StarLightCurves", "Crop"]
CONFIGS = ["paper50", "fixed50", "rec30", "fast10"]


def plan(stage, datasets, configs):
    if stage == "A":
        return [(c, d, ",".join(f"0:{s}" for s in range(10))) for d in datasets or SMALL for c in configs]
    if stage == "B":
        return [(c, d, ",".join(f"{r}:0" for r in range(30))) for d in datasets or SMALL for c in configs]
    if stage == "C":
        return [(c, d, "0:0") for d in datasets or LARGE for c in configs]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=["A", "B", "C"])
    parser.add_argument("--out", type=Path, default=HERE / "output/results.jsonl")
    parser.add_argument("--datasets", help="comma-separated subset")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--threads", default="2", help="Numba/OpenMP threads per job (recorded runs used 2)")
    parser.add_argument("--data-dir")
    parser.add_argument("--indices-dir")
    args = parser.parse_args()

    env = {**os.environ, "OMP_NUM_THREADS": args.threads, "NUMBA_NUM_THREADS": args.threads,
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    extra = []
    if args.data_dir:
        extra += ["--data-dir", args.data_dir]
    if args.indices_dir:
        extra += ["--indices-dir", args.indices_dir]
    datasets = args.datasets.split(",") if args.datasets else None
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for stage in args.stages:
        for config, dataset, pairs in plan(stage, datasets, args.configs.split(",")):
            t0 = time.time()
            proc = subprocess.Popen([sys.executable, str(HERE / "run_job.py"), config, dataset, pairs, *extra],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            lines, errors = [], []
            readers = [threading.Thread(target=lambda: lines.extend(proc.stdout)),
                       threading.Thread(target=lambda: errors.extend(proc.stderr))]
            for r in readers:
                r.start()
            code = proc.wait()
            for r in readers:
                r.join()
            with args.out.open("a") as f:
                for line in lines:
                    if line.startswith("{"):
                        rec = json.loads(line)
                        rec["stage"] = stage
                        f.write(json.dumps(rec) + "\n")
                if code != 0:
                    f.write(json.dumps(dict(stage=stage, config=config, dataset=dataset,
                                            status="killed" if code < 0 else "error", exit_code=code,
                                            stderr_tail="".join(errors)[-500:])) + "\n")
            status = "ok" if code == 0 else f"exit {code}"
            print(f"{stage} {config:8s} {dataset:20s} {status:8s} {time.time() - t0:7.0f}s", flush=True)


if __name__ == "__main__":
    main()
