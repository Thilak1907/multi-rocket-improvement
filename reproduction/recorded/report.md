# Section IV reproduction checks

Source: `/home/claude/repo2/reproduction/recorded/results.jsonl`

## A. Original split, paper configuration, 10 kernel seeds (accuracy %)

| dataset | paper | ours_mean | ours_sd | seed_min | seed_max | in_range |
|---|---|---|---|---|---|---|
| ArrowHead | 86.86 | 86.74 | 0.65 | 85.71 | 88.00 | yes |
| Beef | 76.67 | 76.67 | 0.00 | 76.67 | 76.67 | yes |
| Coffee | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | yes |
| ECG200 | 92.00 | 91.60 | 0.52 | 91.00 | 92.00 | yes |
| GunPoint | 100.00 | 100.00 | 0.00 | 100.00 | 100.00 | yes |
| InsectWingbeatSound | 67.27 | 67.23 | 0.24 | 66.82 | 67.58 | yes |
| ItalyPowerDemand | 96.79 | 96.93 | 0.15 | 96.70 | 97.08 | yes |
| OSULeaf | 95.45 | 96.32 | 0.41 | 95.45 | 96.69 | yes |

## B1. Thirty resamples, seed 0, paired with the published 50k folds (accuracy %)

| dataset | folds | ours | paper | diff_pp | lower | equal | higher | wilcoxon_p |
|---|---|---|---|---|---|---|---|---|
| ArrowHead | 30 | 89.429 | 89.524 | -0.095 | 12 | 9 | 9 | 0.345 |
| Beef | 30 | 77.111 | 77.000 | 0.111 | 5 | 19 | 6 | 0.763 |
| Coffee | 30 | 100.000 | 100.000 | 0.000 | 0 | 30 | 0 |  |
| ECG200 | 30 | 89.000 | 89.067 | -0.067 | 11 | 13 | 6 | 0.315 |
| GunPoint | 30 | 99.756 | 99.778 | -0.022 | 1 | 23 | 6 | 0.206 |
| InsectWingbeatSound | 30 | 66.340 | 67.170 | -0.830 | 27 | 0 | 3 | 0.000 |
| ItalyPowerDemand | 30 | 96.388 | 96.450 | -0.062 | 16 | 1 | 13 | 0.246 |
| OSULeaf | 30 | 96.006 | 96.171 | -0.165 | 17 | 8 | 5 | 0.006 |

Mean over datasets: ours 89.25, paper 89.39.

## B2. 10k configuration against the published 10k folds (accuracy %)

| dataset | ours_10k | paper_10k |
|---|---|---|
| ArrowHead | 89.24 | 89.37 |
| Beef | 74.22 | 77.00 |
| Coffee | 100.00 | 99.76 |
| ECG200 | 89.07 | 88.87 |
| GunPoint | 99.64 | 99.73 |
| InsectWingbeatSound | 66.60 | 67.14 |
| ItalyPowerDemand | 96.46 | 96.45 |
| OSULeaf | 95.76 | 95.73 |

## B3. Configurations paired with the paper configuration (8 datasets x 30 resamples)

Delta is the mean of per-dataset differences; the CI is a dataset bootstrap (10,000 draws, seed 0).

| config | accuracy | delta_pp | ci_low | ci_high | time_s | speedup |
|---|---|---|---|---|---|---|
| paper50 | 89.25 | 0.00 | 0.00 | 0.00 | 426.87 | 1.00 |
| fixed50 | 89.24 | -0.01 | -0.05 | 0.02 | 430.85 | 0.99 |
| rec30 | 89.41 | 0.16 | 0.02 | 0.33 | 240.79 | 1.77 |
| fast10 | 88.87 | -0.38 | -1.13 | 0.07 | 81.11 | 5.26 |

## B4. Effect of the first-difference fix (fixed50 minus paper50, 30 resamples)

| dataset | diff_pp | wilcoxon_p |
|---|---|---|
| ArrowHead | 0.057 | 0.535 |
| Beef | -0.111 | 0.763 |
| Coffee | 0.000 |  |
| ECG200 | 0.000 | 0.976 |
| GunPoint | -0.044 | 0.180 |
| InsectWingbeatSound | -0.044 | 0.356 |
| ItalyPowerDemand | 0.029 | 0.672 |
| OSULeaf | 0.014 | 0.879 |

Empty p-value: identical accuracy on every fold.

## B5. Where the time goes (paper configuration, all stage-B runs)

- Prediction (almost all of it the test-set transform): 70.5% of fit+predict time
- Classifier fitting: 9.9%

## C. Large datasets, original split, seed 0

Exit -9 is SIGKILL; on the recorded 8 GB machine the kernel log showed out-of-memory kills.

| dataset | config | accuracy | paper_r0 | time_s | peak_rss_gib | status |
|---|---|---|---|---|---|---|
| Crop | fast10 | 76.86 | 77.48 | 70.64 | 1.88 | ok |
| Crop | fixed50 |  | 77.48 |  |  | killed (exit -9) |
| Crop | paper50 |  | 77.48 |  |  | killed (exit -9) |
| Crop | rec30 | 77.20 | 77.48 | 100.59 | 3.29 | ok |
| StarLightCurves | fast10 | 98.17 | 98.17 | 14.66 | 1.01 | ok |
| StarLightCurves | fixed50 | 98.13 | 98.17 | 75.79 | 5.94 | ok |
| StarLightCurves | paper50 | 98.14 | 98.17 | 76.06 | 5.95 | ok |
| StarLightCurves | rec30 | 98.14 | 98.17 | 44.22 | 1.33 | ok |
