# Section IV reproduction checks

This folder reproduces the reproduction results in Section IV of the report:
the seed study on the original split, the fold-by-fold comparison with the
published 50k results over 30 resamples, the effect of the first-difference fix,
the runtime breakdown and the memory runs on StarLightCurves and Crop. It also
reruns the four configurations side by side on a second machine.

The fold-by-fold comparison here covers eight datasets. It does not support a
full-archive accuracy claim.

## Recorded machine

These checks ran on a Linux VM reporting an Intel Xeon at 2.30 GHz, with two
allocated vCPUs and an 8 GB cgroup memory limit. The limit is the workload's
memory allowance, not installed host RAM. Numba/OpenMP used two threads and
OpenBLAS/MKL one. The stack was Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, Numba 0.68.1, pandas 3.0.5 and CPU-only Torch; its exact
Torch version was not recorded. See [environment details](recorded/environment.txt).

This is separate from the main Apple M5 benchmark (10 physical cores, 10 logical
CPUs, 24 GiB RAM; four Numba/Torch threads and one BLAS thread). On the Linux
machine, **both `paper50` (legacy) and `fixed50` (corrected)** were killed on
Crop, with OOM entries in the supplied kernel log. These are not the main
benchmark's 9 GiB supervisor-guard stops on ElectricDevices.

## What is here

| File | Purpose |
|---|---|
| `run_job.py` | Runs one configuration on one dataset for a list of (resample, seed) pairs in its own process, and prints accuracy, timings and peak RSS as JSON lines. |
| `run_all.py` | Runs stages A, B and C (below) and appends every result to a JSONL file. A job killed by the operating system is recorded with its exit code. |
| `analyze.py` | Builds the Markdown report from a JSONL file and the authors' published per-fold results. |
| `recorded/results.jsonl` | The recorded runs used in the report (1,358 lines). |
| `recorded/report.md` | `analyze.py` output for the recorded runs. |
| `recorded/published_resample0.csv` | The authors' published resample-0 accuracies for the ten datasets used here (column `PPV+LSPV+MPV+MIPV_base+1st_50k` of `results/results_128_resample_0.csv` at upstream commit `3ccaa4f`). |
| `recorded/environment.txt` | Machine, library versions, commits and data sources of the recorded runs. |
| `recorded/crop_oom_kernel_log.txt` | Kernel log lines for the two 50k runs on Crop that were killed for running out of memory. |

## Configurations

| Name | Settings |
|---|---|
| `paper50` | Published configuration: 50k features, scikit-learn `RidgeClassifierCV`, original (legacy) first difference |
| `fixed50` | `paper50` with the corrected first difference |
| `rec30` | Retained configuration: 30k features, blocked ridge, buffered extraction, in-place processing, batches of 256, folded scaling, corrected first difference |
| `fast10` | `rec30` with a 10k budget |

## Stages

| Stage | Datasets | Runs | Time on 2 cores |
|---|---|---|---|
| A | ArrowHead, Beef, Coffee, ECG200, GunPoint, InsectWingbeatSound, ItalyPowerDemand, OSULeaf | original split, seeds 0-9, 4 configurations | about 10 min |
| B | the same 8 datasets | resamples 0-29, seed 0, 4 configurations | about 25 min |
| C | StarLightCurves, Crop | original split, seed 0, 4 configurations | about 10 min |

## How to run

From the repository root, after installing the requirements:

```bash
# 1. Resample indices and the authors' published results (into data/ and results/)
python scripts/download_data.py --upstream
# 2. UCR archive (into data/UCRArchive_2018)
python scripts/download_data.py --ucr

# 3a. Rebuild the report from the recorded runs (no data or training needed beyond step 1)
python reproduction/analyze.py
# -> reproduction/output/report.md

# 3b. Or rerun everything and analyze the new runs
python reproduction/run_all.py A B C
python reproduction/analyze.py --results reproduction/output/results.jsonl

# Smaller example: one dataset, two configurations
python reproduction/run_all.py B --datasets Coffee --configs paper50,rec30
```

On Windows PowerShell, use the same commands with `python` and backslashes in paths.

## Expected results (recorded runs)

**Original split, 10 seeds:** every published resample-0 accuracy lies inside the
range of our 10 seeds. The sample standard deviation reaches 0.65 points
(ArrowHead, whose seed range is 85.71%–88.00%).

**30 resamples, paired fold by fold with the published 50k folds:**

| Dataset | Ours (%) | Paper (%) | Difference (pp) | Lower / equal / higher | Wilcoxon p |
|---|---|---|---|---|---|
| ArrowHead | 89.43 | 89.52 | -0.10 | 12 / 9 / 9 | 0.35 |
| Beef | 77.11 | 77.00 | +0.11 | 5 / 19 / 6 | 0.76 |
| Coffee | 100.00 | 100.00 | 0.00 | 0 / 30 / 0 | - |
| ECG200 | 89.00 | 89.07 | -0.07 | 11 / 13 / 6 | 0.32 |
| GunPoint | 99.76 | 99.78 | -0.02 | 1 / 23 / 6 | 0.21 |
| InsectWingbeatSound | 66.34 | 67.17 | -0.83 | 27 / 0 / 3 | <0.001 |
| ItalyPowerDemand | 96.39 | 96.45 | -0.06 | 16 / 1 / 13 | 0.25 |
| OSULeaf | 96.01 | 96.17 | -0.17 | 17 / 8 / 5 | 0.006 |
| **Mean** | **89.25** | **89.39** | **-0.14** | | |

**First-difference fix:** `fixed50` minus `paper50` is -0.01 points on average;
no dataset has Wilcoxon p below 0.17.

**Time:** with the paper configuration, prediction (almost all of it the test-set
transform) takes 70.5% of fit-plus-predict time and classifier fitting 9.9%.

**All four configurations, 8 datasets x 30 resamples:**

| Configuration | Accuracy (%) | Change (pp) [95% CI] | Time (s) | Speed-up |
|---|---|---|---|---|
| paper50 | 89.25 | - | 426.9 | 1.00x |
| fixed50 | 89.24 | -0.01 [-0.05, +0.02] | 430.8 | 0.99x |
| rec30 | 89.41 | +0.16 [+0.02, +0.33] | 240.8 | 1.77x |
| fast10 | 88.87 | -0.38 [-1.13, +0.07] | 81.1 | 5.26x |

The +0.16 points for `rec30` comes mostly from Beef and OSULeaf and does not hold on
the 55-dataset benchmark, so it is not claimed as a gain.

**Large datasets, original split (8 GB memory limit):**

The StarLightCurves `paper50` peak is **6.4 GB (5.95 GiB)**. GB uses decimal
bytes (10^9); GiB uses binary bytes (2^30). The generated report includes both
units. The recorded 6,088.72 MiB converts to 6.4 GB when rounded to one decimal.

| Dataset | Configuration | Accuracy (%) | Published (%) | Time (s) | Peak RSS (GiB) |
|---|---|---|---|---|---|
| StarLightCurves | paper50 | 98.14 | 98.17 | 76.1 | 5.95 |
| StarLightCurves | rec30 | 98.14 | 98.17 | 44.2 | 1.33 |
| StarLightCurves | fast10 | 98.17 | 98.17 | 14.7 | 1.01 |
| Crop | paper50, fixed50 | killed (out of memory) | 77.48 | - | - |
| Crop | rec30 | 77.20 | 77.48 | 100.6 | 3.29 |
| Crop | fast10 | 76.86 | 77.48 | 70.6 | 1.88 |

## Notes

- Results are deterministic for a given seed, machine and library versions; 20
  stage-A runs repeated at the current commit gave identical accuracies. Timings
  and memory depend on the hardware and thread settings.
- Four datasets in the recorded runs (ArrowHead, GunPoint, ItalyPowerDemand,
  OSULeaf) came from sktime's bundled copies rather than the official archive ZIP
  (see `recorded/environment.txt`). Reruns on the official TSVs may differ in the
  last digit on those datasets.
- Peak RSS includes imported libraries (about 0.8 GiB with PyTorch).
- Whether the 50k configuration is killed on Crop depends on available memory; on
  machines with more than about 8 GB it should finish.
- Analysis rejects duplicate successful runs and incomplete Stage B comparisons.
  `run_all.py` appends to its output file; use a new `--out` path for an independent
  rerun instead of appending the same jobs twice. A full Stage B comparison needs
  the same 30 resamples for every selected configuration. Single-configuration
  memory-only Stage C files can be analysed without Stage B.
