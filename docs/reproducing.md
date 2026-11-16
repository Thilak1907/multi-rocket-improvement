# Reproducing the experiments

The root README covers setup and downloads. This guide distinguishes rebuilding
recorded results from generating new measurements. Both operations leave the
checked-in reports unchanged.

## 1. Environment and inputs

Use Python 3.11 and `pip install -r requirements-reproduce.txt`. The main machine
is an Apple M5 (Mac17,3), with 10 physical cores, 10 logical CPUs and 24 GiB RAM,
running macOS 26.5.2. The benchmark uses four Numba/Torch threads and one BLAS thread.
Linux or other CPUs can change timing, RSS, and floating-point behavior. Exact
numeric-package versions and measurement conventions are recorded in
[`environment.json`](../reproducibility/environment.json).

The separate Section IV checks use the recorded two-vCPU Intel Xeon Linux VM
at 2.30 GHz with an 8 GB cgroup memory limit. They use two Numba/OpenMP threads,
one OpenBLAS/MKL thread and a different Python/library stack, documented in
[`reproduction/recorded/environment.txt`](../reproduction/recorded/environment.txt).
Both legacy and corrected 50k Crop attempts were OOM-killed there. Those kills
are separate from the main macOS benchmark's 9 GiB guard stops on ElectricDevices.

```bash
python scripts/download_data.py --upstream --ucr
python scripts/reproduce.py verify
```

The official UCR ZIP is password protected. Read the
[archive briefing](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and enter
its public password when prompted. The downloader extracts only the 109 required
fixed-length datasets and checks all 218 TSV hashes. It also installs resample
indices and metadata from upstream commit
`3ccaa4f8ed11769b904af885e17018c39c89a2ac`, with archive and per-file SHA-256 checks.

For an existing ZIP, pass `--ucr-archive /path/to/UCRArchive_2018.zip`. For an
existing dataset directory, use `scripts/reproduce.py verify --data-dir PATH`
and `run ... --data-dir PATH`; indices still belong under `data/indices109/`.
Do not convert TSVs through float32 or regenerate splits: those changes can alter
inputs and break the paired comparison.

## 2. Rebuild the recorded results

```bash
python scripts/reproduce.py summarize all
python scripts/compare_paper.py
```

No dataset download is needed for this operation. The runner verifies the
supplied ZIP checksums, extracts the recorded JSON/CSV/prediction records and
analysis code into isolated workspaces, and regenerates tables and figures.
This validates analysis of existing measurements; it is not a new training run.

The regenerated reports are under:

```text
output/reproductions/recorded/<profile>/reports/<report-name>/
output/paper_comparison/
```

Use `--workspace /path/to/empty-directory` with a single profile to choose
another location. Existing workspaces must have the matching profile and artifact
checksums. The tool does not reuse recorded measurement files for a fresh rerun.

## 3. Generate new measurements

```bash
python scripts/reproduce.py run review --smoke
python scripts/reproduce.py run review
```

The smoke test runs ArrowHead only and is not an archive-level result. The full
review runs 2,021 jobs, including resource-guard attempts. There is one worker
process per job, with warm JIT compilation before measured work. Complete
training includes feature extraction, cleaning/scaling, and ridge fitting.
Complete prediction includes feature extraction and classification. Imports,
data loading, and warmup are excluded from timings but included in process RSS.

Fresh runs go under `output/reproductions/rerun/<profile>/`. Source archives
contain the runtime versions used for the corresponding study, so subsequent
changes to the working model do not silently alter a historical rerun. Existing
job records allow a stopped experiment to resume. To start another independent
run, choose a new `--workspace`.

### Profiles and commands

| Profile | Training command | Analysis command |
| --- | --- | --- |
| Corrected review | `python scripts/reproduce.py run review` | `python scripts/reproduce.py summarize review` |
| Additional-view ensemble | `python scripts/reproduce.py run accuracy` | `python scripts/reproduce.py summarize accuracy` |
| Fixed feature maps | `python scripts/reproduce.py run feature-power` | `python scripts/reproduce.py summarize feature-power` |
| Expanded ridge grid | `python scripts/reproduce.py run ridge-range` | `python scripts/reproduce.py summarize ridge-range` |
| Legacy configuration search | `python scripts/reproduce.py run legacy-combinations` | `python scripts/reproduce.py summarize legacy-combinations` |
| Early optimization comparison | `python scripts/reproduce.py run legacy-optimization --allow-equivalent` | `python scripts/reproduce.py summarize legacy-optimization` |
| Initial repeated optimization run | `python scripts/reproduce.py run legacy-initial --allow-equivalent` | `python scripts/reproduce.py summarize legacy-initial` |

The last two profiles have an important limit: their exact early model sources
were not snapshotted. Analysis reproduces their retained measurements. Training
uses a later legacy implementation with the same settings and is explicitly
marked an equivalent-settings rerun, not exact source replay. The manifest lists
the missing original sources. Do not label newly measured times as the original
published project times.

### Which command produces each main result?

`run review` invokes the pinned `experiments.review_revision` worker and its
report generator. It includes:

| Result in the corrected report | Stage | Generated CSV/figure |
| --- | --- | --- |
| Component costs and folding decision | 12 selection datasets, capped TRAIN-only validation, 10 profiles, 3 repeats | `ablation_summary.csv`, `ablation.png` |
| Six-finalist timing/accuracy table | 55 datasets, resample 1, 3 timing repeats | `finalists.csv`, `per_dataset.csv` |
| Stage times, feature memory, model size | Same finalist jobs | `stage_times_and_memory.csv` |
| Twelve-configuration trade-off plot | Common first timing run on 55 datasets | `all_configurations.csv`, `tradeoffs_all.svg` |
| Accuracy interval and non-inferiority | Baseline/fixed/routed, resamples 1–3 | `noninferiority.csv`, `paired_resample_differences.csv` |
| Alpha and prediction agreement | Paired 30k comparisons | `ridge_parity.csv` |
| ElectricDevices memory behavior | One attempt per finalist, 9 GiB guard | `electricdevices.csv`, `failures.csv` |
| Legacy replay | Five recorded reference comparisons | `legacy_replay.csv` |
| Size-routing adoption decision | Frozen gates over the above stages | `decision.json` |

The trade-off plot uses its original layout and configuration IDs. `fast30`
is the retained model, while the starred `auto30` is the routing experiment.
The dotted horizontal line is baseline accuracy, the vertical line is 1× speed,
and the dashed line is the accuracy/speed Pareto frontier. Memory is not included
in that frontier. The figure is generated by `summarize_revision.py`.

The accuracy profile selects one of three extra views on 54 TRAIN-only selection
partitions, then evaluates only the frozen winner on 55 datasets and resamples
1–3. It writes `selection_summary.csv`, `confirmation_summary.csv`,
`paired_differences.csv`, and `accuracy_comparison.svg`.

The feature-power and ridge-range profiles use the same 54-name TRAIN-only
selection screen. Neither passed its gate in the recorded runs, so they have no
confirmation result. A fresh run may proceed to confirmation if its selection
gate passes. That is a new experiment, not a replacement for the recorded result.

The legacy-combinations profile runs the original search, repeated confirmation,
fixed-30k refinement, and stress follow-ups in sequence. Choices that depend on
measured time can differ on another machine; retain those differences rather
than forcing the original winner.

## 4. Compare with the paper

```bash
python scripts/compare_paper.py
```

This aggregates the authors' released 50k CSVs over both the full 109 names and
the project's 55 confirmation names, and reads project accuracy from the
retained per-resample CSV. It writes `comparison.csv` and `comparison.md` under
`output/paper_comparison/`. The checked-in copy is under
[`reports/paper_comparison`](../reports/paper_comparison/comparison.md).

The 55-name published subset is an aggregation of the released numbers, not a
table printed in the article. The paper's 30-resample accuracy, its 112-problem
single-thread timing table, the project's one-resample timing comparison, and
the project's three-resample accuracy test are different protocols. In
particular, the project speed-up is measured against a fresh local baseline,
not against the paper's hardware.

## 5. Check the result, not just the command exit status

For a rerun, inspect the report's job counts, failure records, and decision.
A dataset stopped by the polling guard should remain in `failures.csv`; it must
not disappear from a denominator. Timing repetitions reuse the same split and
seed and must not count as extra accuracy resamples.

The efficiency analysis averages paired accuracy differences across three
resamples within each of 55 datasets, then uses those dataset means for its
bootstrap and non-inferiority test. The chosen margin is −0.3 percentage points.
The optional accuracy-addition experiments require an interval above zero;
passing the efficiency margin does not establish accuracy superiority.

The historical confirmation names were inspected during earlier development.
These follow-up measurements are not untouched external validation. Resource
measurements exclude neither all system variability nor hardware effects.

## 6. Artifact layout

[`reproducibility/manifest.json`](../reproducibility/manifest.json) lists the
source bundles, individual file hashes, source fidelity, and original protocol
hashes. `records.zip` contains measurement records, prediction arrays needed for
agreement checks, metadata, and author result CSVs. It excludes datasets,
virtual environments, logs, and caches.

Absolute local paths in JSON have been replaced with `${REPO}`. The runner
binds them to the isolated workspace and the chosen data directory. Consequently,
portable protocol file hashes differ from the original local protocol hashes;
those original hashes remain in the manifest for provenance. Prediction hashes,
measurements, source bytes, and the model's dataset/index hashes are retained.

If a source or input hash changes, use a new workspace. Do not edit an old
protocol just to bypass a mismatch. The frozen source bundles retain the
repository's GPL-3.0 attribution and the vendored HYDRA notice.
