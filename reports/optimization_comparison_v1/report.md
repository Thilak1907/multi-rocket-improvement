# MultiRocket optimization comparison

Selected pair: **LSPV+MPV**, chosen on all 54 selection datasets.
Held-out measurements completed: **990/990** (55 datasets, 3 timing repetitions, six variants).
Only complete dataset/variant pairs enter each summary; counts are shown. Timing repetitions reuse the same seed and split, so they are not independent accuracy resamples.

| variant | datasets | accuracy | delta_pp | wins | ties | losses | classifier_speedup | total_speedup | median_rss_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 55 | 85.726% | +0.000 | 0 | 55 | 0 | 1.00x | 1.00x | 0.0% |
| pooling_only | 55 | 85.455% | -0.271 | 14 | 12 | 29 | 1.78x | 0.98x | 26.5% |
| ridge_only | 55 | 85.726% | +0.000 | 0 | 55 | 0 | 1.23x | 1.04x | 11.6% |
| memory_only | 55 | 85.726% | +0.000 | 0 | 55 | 0 | 1.09x | 1.04x | 2.5% |
| pooling_ridge | 55 | 85.455% | -0.271 | 14 | 12 | 29 | 1.71x | 0.98x | 34.6% |
| combined | 55 | 85.455% | -0.271 | 14 | 12 | 29 | 1.70x | 0.98x | 42.1% |

Speedups are ratios of summed per-dataset median times against fresh seeded baseline runs on the same machine. RSS reduction is the median per-dataset reduction in process peak RSS; it includes imports and inputs. Peak RSS and feature-array size are different measures.

## Runtime and memory by stage

Times below sum per-dataset medians over the same paired datasets in each row. "Other fit" includes kernel generation, differencing and feature cleaning; "other predict" includes differencing, cleaning and batch overhead. Scaling is included in classifier fit/predict. Seconds are shown throughout.

| variant | fit total (s) | train features (s) | scale + ridge fit (s) | other fit (s) |
| --- | --- | --- | --- | --- |
| baseline | 63.99 | 38.03 | 21.64 | 4.32 |
| pooling_only | 58.00 | 42.29 | 12.16 | 3.55 |
| ridge_only | 59.54 | 37.60 | 17.62 | 4.32 |
| memory_only | 61.30 | 37.57 | 19.81 | 3.93 |
| pooling_ridge | 58.78 | 42.54 | 12.66 | 3.59 |
| combined | 58.95 | 42.65 | 12.75 | 3.54 |

| variant | predict total (s) | test features (s) | scale + ridge predict (s) | other predict (s) | fit + predict (s) |
| --- | --- | --- | --- | --- | --- |
| baseline | 87.03 | 81.92 | 1.44 | 3.67 | 151.06 |
| pooling_only | 96.41 | 94.34 | 0.62 | 1.44 | 154.53 |
| ridge_only | 85.15 | 81.29 | 1.24 | 2.62 | 144.73 |
| memory_only | 83.67 | 80.77 | 1.17 | 1.73 | 145.24 |
| pooling_ridge | 96.17 | 94.20 | 0.62 | 1.35 | 154.88 |
| combined | 95.27 | 93.71 | 0.62 | 0.93 | 154.29 |

Feature memory is the median per-dataset feature-array size. Peak RSS is the median total process peak and includes imports, loaded data, and solver workspace.

| variant | features | train features (MiB) | largest test batch (MiB) | peak RSS (GiB) |
| --- | --- | --- | --- | --- |
| baseline | 49728 | 34.34 | 58.43 | 1.03 |
| pooling_only | 24864 | 17.17 | 29.21 | 0.70 |
| ridge_only | 49728 | 34.34 | 58.43 | 0.76 |
| memory_only | 49728 | 34.34 | 48.56 | 0.73 |
| pooling_ridge | 24864 | 17.17 | 29.21 | 0.58 |
| combined | 24864 | 17.17 | 24.28 | 0.51 |

The per-dataset CSV also includes alpha, published and local accuracy, plus the older repo's recorded classifier, transform, kernel-generation, and differencing times. Prediction agreement is in parity.csv. Raw three-repeat records and predictions remain in the experiment output.

## What improved and what changed

Across the 55 paired held-out datasets, all three changes together give **1.70x classifier-fit speedup** and **0.98x end-to-end speedup** (sums of per-dataset median times). Mean accuracy changes by **-0.271 percentage points**; dataset-bootstrap 95% interval **[-0.768, +0.275] pp**.
The reduced transform allocates exactly half as many feature columns. Median per-dataset process peak RSS changes by **-42.1%**. Reducing feature columns does not imply halving total memory or runtime: unchanged convolution work, Python/library memory, and the sample-space eigendecomposition remain.
The isolated rows above distinguish the effects: pooling changes the representation and can change accuracy; the ridge and memory rows test computational changes on the same representation. Prediction/alpha parity is reported below, so equal aggregate accuracy cannot hide changed predictions.

## Protocol and isolation

Nominal feature budget 50,000: all four operators produce 49,728 columns; two produce 24,864. The raw and first-difference kernels, biases, data split, and per-dataset seed are shared across variants.
The fixed split has 54 selection and 55 held-out datasets (split seed 42); the earlier partial run's split was retained but its scores were not reused. Six pairs compete by unweighted mean selection accuracy. The choice and selection-file hashes are frozen before evaluation. Selection uses the fast solver's same LOO-MSE objective; evaluation includes sklearn references.
Resample 0 only, 4 Numba threads, one BLAS thread, warm JIT, prediction batches of 256. Per-job RSS guard: 9.0 GiB. Guarded terminations are reported as resource-limit failures, not OS OOMs. Hardware: macOS-26.5.2-arm64-arm-64bit; sklearn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, Numba 0.67.0.
Variant definitions: baseline = four operators + sklearn + original copying/full prediction; pooling_only = selected pair; ridge_only = fast solver; memory_only = in-place cleaning/scaling + chunked prediction; pooling_ridge = pair + fast solver; combined = all three.

| operators | selection_accuracy |
| --- | --- |
| LSPV+MPV | 87.330672% |
| PPV+LSPV | 87.326552% |
| PPV+MPV | 87.071213% |
| LSPV+MIPV | 85.914886% |
| PPV+MIPV | 85.756712% |
| MPV+MIPV | 84.276442% |

The winning margin over the next pair is only 0.004120 percentage points on the selection half. This is a fragile ranking; the pair is nevertheless kept fixed for the independent held-out comparison. No held-out result is used to revise it.

## Paper and repository context

The attached paper (arXiv:2102.00457v4, sections 3.5, 4.1 and 4.3.2, Table 2, Figure 10) uses 49,728 features by default, 30 resamples for the 109-dataset accuracy comparison, and a separate 112-problem single-resample timing table (15.77 minutes training at 50k). Its hardware was AMD EPYC 7702; the main experiments used 32 threads and 64 GB RAM. Its pooling design was tuned on 40 development datasets, and all four operators ranked best in that ablation.
The per-dataset CSV includes the published fold-0 accuracy, published 30-resample mean, and the existing local result where available. Those contextual columns have different random kernels and, for the mean, different train/test splits. They cannot establish an improvement by themselves; fresh paired runs are the causal comparison. No cross-machine speedup against the paper is claimed.
In the local repository CSVs, train_time measures scaler/classifier fitting, rather than full transformation plus fitting. The new measurements record both classifier_fit_s and complete fit/predict wall-clock times, including feature cleaning.
| reference | datasets | accuracy |
| --- | --- | --- |
| Published MultiRocket, resample 0 | 55 | 86.077% |
| Published MultiRocket, 30-resample mean | 55 | 87.654% |
| Existing local 50k results, resample 0 | 55 | 85.659% |
| Fresh seeded baseline, resample 0 | 55 | 85.726% |
All reference averages above use the same held-out dataset names. Figure 10 is context for operator diversity; it is not a published result for this fixed-kernel, half-feature experiment.
The new fast solver matches the default sklearn leave-one-out squared-error criterion, including the unpenalized intercept. It uses float64 blocked Gram accumulation and fits coefficients only for the selected alpha. For wide matrices, it still requires O(n_train²) memory and O(n_train³) eigendecomposition. Tall matrices fall back to sklearn.
Memory changes remove cleaning/scaling output copies and bound test feature allocation by the batch size. The original class already kept training features local to fit; explicit deletion makes the lifetime clear but is not a new elimination of persistent cached features.

## Alpha and prediction parity

| variant | reference | datasets | alpha_matches | minimum_prediction_agreement |
| --- | --- | --- | --- | --- |
| combined | pooling_only | 56 | 55 | 100.000000% |
| memory_only | baseline | 55 | 55 | 100.000000% |
| pooling_ridge | pooling_only | 56 | 55 | 100.000000% |
| ridge_only | baseline | 55 | 55 | 100.000000% |

These are observed parity results, not a universal guarantee of bitwise identity on every numerical edge case. Unit checks also cover binary/multiclass targets, rank deficiency, constant columns, constant matrices, float32/float64, and the tall-data sklearn fallback.

## ElectricDevices stress case

ElectricDevices belongs to the selection half, so it is excluded from held-out accuracy aggregates. An existing successful local 50k run already exists; an historical OOM is not independently established by the available artifacts.
| variant | accuracy | features | total_s | classifier_fit_s | peak_rss_bytes | train_feature_bytes | max_prediction_feature_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combined | 72.935% | 24864.0 | 71.582 | 63.903 | 5.503 GiB | 846.6 MiB | 24.3 MiB |
| memory_only | 73.739% | 49728.0 | 80.147 | 72.316 | 7.459 GiB | 1693.2 MiB | 48.6 MiB |
| pooling_only | 72.935% | 24864.0 | 75.480 | 66.629 | 5.899 GiB | 846.6 MiB | 731.4 MiB |
| pooling_ridge | 72.935% | 24864.0 | 71.624 | 63.934 | 5.920 GiB | 846.6 MiB | 731.4 MiB |
| ridge_only | 73.739% | 49728.0 | 82.540 | 74.523 | 7.801 GiB | 1693.2 MiB | 1462.8 MiB |

The table above summarizes successful runs only. Every attempt, including guard stops, is counted here. A guard stop means the 9 GiB process limit was crossed; it does not establish an operating-system out-of-memory crash.

| variant | attempts | successful | rss_limit | other_failures | largest_observed_rss_gib |
| --- | --- | --- | --- | --- | --- |
| baseline | 3 | 0 | 3 | 0 | 9.19 GiB |
| combined | 3 | 3 | 0 | 0 | 5.50 GiB |
| memory_only | 3 | 1 | 2 | 0 | 9.08 GiB |
| pooling_only | 3 | 1 | 2 | 0 | 9.16 GiB |
| pooling_ridge | 3 | 3 | 0 | 0 | 5.92 GiB |
| ridge_only | 3 | 1 | 2 | 0 | 9.08 GiB |

Published and earlier local ElectricDevices accuracy is contextual; the old local fit time records scaler/classifier fit only and has no peak-RSS measurement.

| reference | accuracy | classifier_fit_s |
| --- | --- | --- |
| Paper/repo published resample 0 | 74.024% | — |
| Paper/repo published 30-resample mean | 90.035% | — |
| Existing local 50k run | 73.752% | 64.43 |

## Dataset-level gains and losses

Five lowest and five highest paired accuracy changes for the combined variant (see per_dataset.csv for every dataset):

| dataset | baseline_accuracy | accuracy | delta_pp | total_speedup |
| --- | --- | --- | --- | --- |
| SemgHandMovementCh2 | 78.000% | 71.111% | -6.889 | 0.94x |
| CinCECGTorso | 94.783% | 90.435% | -4.348 | 0.86x |
| SemgHandSubjectCh2 | 91.778% | 88.222% | -3.556 | 0.91x |
| DistalPhalanxOutlineCorrect | 80.435% | 77.899% | -2.536 | 1.27x |
| CricketY | 85.385% | 83.077% | -2.308 | 1.01x |
| Rock | 84.000% | 86.000% | +2.000 | 0.88x |
| Computers | 76.800% | 79.200% | +2.400 | 0.93x |
| WormsTwoClass | 79.221% | 81.818% | +2.597 | 0.92x |
| Lightning2 | 68.852% | 73.770% | +4.918 | 0.95x |
| FaceAll | 80.118% | 87.160% | +7.041 | 1.07x |

## Limits and reproducibility

This is a single train/test split per dataset, not the paper's full 30-resample replication. Dataset-bootstrap confidence intervals for mean paired accuracy differences are in summary.json; they describe variability across this held-out collection, not across training resamples. Pair selection and timings may vary with seeds, versions, and hardware.
The frozen protocol, dataset/source SHA-256 hashes, selected pair, all raw timing/RSS records, predictions, and worker logs are in `output/optimization_comparison/50k_split42`. Per-dataset, stage, parity, ElectricDevices and summary CSV files are beside this report.
Run: `.venv/bin/python -m experiments.compare_optimizations` (resumes matching completed jobs); then `.venv/bin/python -m experiments.summarize_optimizations`. Use a new output directory after source changes. Run tests with `.venv/bin/python -m unittest discover -s tests -v`.

Failures: [{"dataset": "ElectricDevices", "variant": "baseline", "repeat": 0, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9868722176, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "baseline", "repeat": 1, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9678503936, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "baseline", "repeat": 2, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9845211136, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "memory_only", "repeat": 1, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9754591232, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "memory_only", "repeat": 2, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9754624000, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "pooling_only", "repeat": 1, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9733455872, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "pooling_only", "repeat": 2, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9840050176, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "ridge_only", "repeat": 1, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9722986496, "prediction_hash": NaN, "exitcode": -9.0}, {"dataset": "ElectricDevices", "variant": "ridge_only", "repeat": 2, "status": "rss_limit", "seed": NaN, "accuracy": NaN, "alpha": NaN, "features": NaN, "train_feature_bytes": NaN, "max_prediction_feature_bytes": NaN, "fit_s": NaN, "predict_s": NaN, "total_s": NaN, "classifier_fit_s": NaN, "classifier_predict_s": NaN, "train_transform_s": NaN, "test_transform_s": NaN, "peak_rss_bytes": 9754771456, "prediction_hash": NaN, "exitcode": -9.0}]
