> **Legacy results:** these runs preserve the original first-difference alignment and do not include the L−1 correction. The corrected, size-based-ridge review is recorded separately in `reports/review_revision`. Previous figures remain available for comparison.

# MultiRocket optimization comparison

Selected pair: **LSPV+MPV**, chosen on all 54 selection datasets.
Held-out measurements completed: **330/330** (55 datasets, 1 timing repetition, six variants).
Only complete dataset/variant pairs enter each summary; counts are shown. Timing repetitions reuse the same seed and split, so they are not independent accuracy resamples.

| variant | datasets | accuracy | delta_pp | wins | ties | losses | classifier_speedup | total_speedup | median_rss_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 55 | 85.726% | +0.000 | 0 | 55 | 0 | 1.00x | 1.00x | 0.0% |
| pooling_only | 55 | 85.455% | -0.271 | 14 | 12 | 29 | 1.57x | 0.95x | 31.0% |
| ridge_only | 55 | 85.726% | +0.000 | 0 | 55 | 0 | 0.99x | 1.00x | 0.0% |
| memory_only | 55 | 85.726% | +0.000 | 0 | 55 | 0 | 1.04x | 1.02x | 2.4% |
| pooling_ridge | 55 | 85.455% | -0.271 | 14 | 12 | 29 | 1.42x | 0.94x | 31.9% |
| combined | 55 | 85.455% | -0.271 | 14 | 12 | 29 | 1.42x | 0.95x | 32.0% |

Speedups are ratios of summed per-dataset median times against fresh seeded baseline runs on the same machine. RSS reduction is the median per-dataset reduction in process peak RSS; it includes imports and inputs. Peak RSS and feature-array size are different measures.

The archived three-repetition run on the same held-out split provides timing context; its fast ridge alpha differed from sklearn on one near tie, although predictions matched. The corrected single-repetition run above resolves that alpha choice. Small timing ratios vary between runs, so treat marginal speed changes cautiously. See `../optimization_comparison_v1/report.md`.

## Runtime and memory by stage

Times below sum per-dataset medians over the same paired datasets in each row. "Other fit" includes kernel generation, differencing and feature cleaning; "other predict" includes differencing, cleaning and batch overhead. Scaling is included in classifier fit/predict. Seconds are shown throughout.

| variant | fit total (s) | train features (s) | scale + ridge fit (s) | other fit (s) |
| --- | --- | --- | --- | --- |
| baseline | 64.74 | 40.13 | 20.27 | 4.34 |
| pooling_only | 62.35 | 45.83 | 12.87 | 3.65 |
| ridge_only | 65.04 | 40.22 | 20.48 | 4.33 |
| memory_only | 63.99 | 40.48 | 19.48 | 4.04 |
| pooling_ridge | 63.83 | 45.89 | 14.26 | 3.68 |
| combined | 63.43 | 45.58 | 14.32 | 3.53 |

| variant | predict total (s) | test features (s) | scale + ridge predict (s) | other predict (s) | fit + predict (s) |
| --- | --- | --- | --- | --- | --- |
| baseline | 92.15 | 88.23 | 1.37 | 2.54 | 156.88 |
| pooling_only | 103.45 | 101.44 | 0.67 | 1.34 | 165.80 |
| ridge_only | 91.91 | 88.00 | 1.27 | 2.64 | 156.94 |
| memory_only | 90.46 | 87.27 | 1.29 | 1.90 | 154.45 |
| pooling_ridge | 103.39 | 101.39 | 0.66 | 1.34 | 167.23 |
| combined | 102.03 | 100.31 | 0.67 | 1.05 | 165.45 |

Feature memory is the median per-dataset feature-array size. Peak RSS is the median total process peak and includes imports, loaded data, and solver workspace.

| variant | features | train features (MiB) | largest test batch (MiB) | peak RSS (GiB) |
| --- | --- | --- | --- | --- |
| baseline | 49728 | 34.34 | 58.43 | 1.03 |
| pooling_only | 24864 | 17.17 | 29.21 | 0.70 |
| ridge_only | 49728 | 34.34 | 58.43 | 0.86 |
| memory_only | 49728 | 34.34 | 48.56 | 0.73 |
| pooling_ridge | 24864 | 17.17 | 29.21 | 0.62 |
| combined | 24864 | 17.17 | 24.28 | 0.53 |

The per-dataset CSV also includes alpha, published and local accuracy, plus the older repo's recorded classifier, transform, kernel-generation, and differencing times. Prediction agreement is in parity.csv. Raw 1-repeat records and predictions remain in the experiment output.

## What improved and what changed

Across the 55 paired held-out datasets, all three changes together give **1.42x classifier-fit speedup** and **0.95x end-to-end speedup** (sums of per-dataset median times). Mean accuracy changes by **-0.271 percentage points**; dataset-bootstrap 95% interval **[-0.768, +0.275] pp**.
The reduced transform allocates exactly half as many feature columns. Median per-dataset process peak RSS changes by **-32.0%**. Reducing feature columns does not imply halving total memory or runtime: unchanged convolution work, Python/library memory, and the sample-space eigendecomposition remain.
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
| combined | pooling_only | 56 | 56 | 100.000000% |
| memory_only | baseline | 55 | 55 | 100.000000% |
| pooling_ridge | pooling_only | 56 | 56 | 100.000000% |
| ridge_only | baseline | 55 | 55 | 100.000000% |

These are observed parity results, not a universal guarantee of bitwise identity on every numerical edge case. Unit checks also cover binary/multiclass targets, rank deficiency, constant columns, constant matrices, float32/float64, and the tall-data sklearn fallback.

## ElectricDevices stress case

ElectricDevices belongs to the selection half, so it is excluded from held-out accuracy aggregates. An existing successful local 50k run already exists; an historical OOM is not independently established by the available artifacts.
| variant | accuracy | features | total_s | classifier_fit_s | peak_rss_bytes | train_feature_bytes | max_prediction_feature_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combined | 72.935% | 24864.0 | 75.387 | 66.599 | 5.503 GiB | 846.6 MiB | 24.3 MiB |
| memory_only | 73.739% | 49728.0 | 83.031 | 74.759 | 6.702 GiB | 1693.2 MiB | 48.6 MiB |
| pooling_only | 72.935% | 24864.0 | 74.819 | 65.523 | 6.377 GiB | 846.6 MiB | 731.4 MiB |
| pooling_ridge | 72.935% | 24864.0 | 75.379 | 66.779 | 5.920 GiB | 846.6 MiB | 731.4 MiB |
| ridge_only | 73.739% | 49728.0 | 82.291 | 73.584 | 7.497 GiB | 1693.2 MiB | 1462.8 MiB |

The table above summarizes successful runs only. Every attempt, including guard stops, is counted here. A guard stop means the 9 GiB process limit was crossed; it does not establish an operating-system out-of-memory crash.

| variant | attempts | successful | rss_limit | other_failures | largest_observed_rss_gib |
| --- | --- | --- | --- | --- | --- |
| baseline | 1 | 0 | 1 | 0 | 9.27 GiB |
| combined | 1 | 1 | 0 | 0 | 5.50 GiB |
| memory_only | 1 | 1 | 0 | 0 | 6.70 GiB |
| pooling_only | 1 | 1 | 0 | 0 | 6.38 GiB |
| pooling_ridge | 1 | 1 | 0 | 0 | 5.92 GiB |
| ridge_only | 1 | 1 | 0 | 0 | 7.50 GiB |

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
| CinCECGTorso | 94.783% | 90.435% | -4.348 | 0.87x |
| SemgHandSubjectCh2 | 91.778% | 88.222% | -3.556 | 0.91x |
| DistalPhalanxOutlineCorrect | 80.435% | 77.899% | -2.536 | 1.35x |
| CricketY | 85.385% | 83.077% | -2.308 | 0.97x |
| Rock | 84.000% | 86.000% | +2.000 | 0.86x |
| Computers | 76.800% | 79.200% | +2.400 | 0.82x |
| WormsTwoClass | 79.221% | 81.818% | +2.597 | 0.81x |
| Lightning2 | 68.852% | 73.770% | +4.918 | 0.94x |
| FaceAll | 80.118% | 87.160% | +7.041 | 1.05x |

## Limits and reproducibility

This is a single train/test split per dataset, not the paper's full 30-resample replication. Dataset-bootstrap confidence intervals for mean paired accuracy differences are in summary.json; they describe variability across this held-out collection, not across training resamples. Pair selection and timings may vary with seeds, versions, and hardware.
The frozen protocol, dataset/source SHA-256 hashes, selected pair, all raw timing/RSS records, predictions, and worker logs are in `output/optimization_comparison/50k_split42_alpha_tie`. Per-dataset, stage, parity, ElectricDevices and summary CSV files are beside this report.
Run: `.venv/bin/python -m experiments.compare_optimizations --output output/optimization_comparison/50k_split42_alpha_tie --repeats 1` (resumes matching completed jobs); then `.venv/bin/python -m experiments.summarize_optimizations --output output/optimization_comparison/50k_split42_alpha_tie`. Use a new output directory after source changes. Run tests with `.venv/bin/python -m unittest discover -s tests -v`.

Job failures: ElectricDevices baseline repeat 0: rss_limit Full records are in electricdevices_status.csv and the raw experiment output.
