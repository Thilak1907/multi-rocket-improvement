# Expanded ridge regularization range for the retained 30k bundle

The production default remains unchanged by this experiment. Every model keeps the corrected 30k bundle, all four pooling operators, blocked ridge, buffered extraction, in-place processing, 256-row prediction batches, and folded scaling. Each candidate has 29,568 columns and one ridge head, without an ensemble.

The sole change is expanding the ridge alpha grid. Keep the original `np.logspace(-3,3,10)` and add `[1e-6,1e-5,1e-4,1e4,1e5,1e6]`, for 16 candidates. Kernel seeds, features, scaling, leave-one-out squared-error objective, and prediction code remain unchanged. This allows stronger or weaker regularization when the default grid selects an endpoint.

Background: alpha is the ridge regularization strength; extending the search range can change the chosen model but does not guarantee improved accuracy ([sklearn RidgeClassifierCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)).

## Frozen selection

54 selection dataset names; stratified TRAIN-only 80/20 partitions, capped at 512 cases in each partition. Choose the largest mean validation gain, provided it is at least +0.1 percentage points and summed fit-plus-predict runtime is at most 1.20 times baseline. Candidates are not selected using confirmation labels.

| Grid | Validation accuracy | Change | Runtime / baseline |
| --- | ---: | ---: | ---: |
| expanded | 84.317% | -1.038 pp | 0.987× |

**The expanded grid did not pass selection. Retain the current recommendation.**

## Limits and reproduction

One timing measurement per split; small runtime differences are descriptive. Alpha-grid search is included in fit/predict time. Baseline is shared across comparisons and evaluated first; candidate order is rotated. Process high-water RSS includes imports/warmup and earlier fits, so it is an upper bound rather than isolated candidate memory. Every dataset worker has a 9 GiB polling guard.

Confirmation names were inspected during earlier development; these are follow-up paired measurements, not untouched external validation. Resamples reuse examples. This is a finite benchmark comparison, not a universal accuracy guarantee. No hardware, energy, multivariate, or GPU improvement is established.

Rebuild the recorded tables with `python scripts/reproduce.py summarize ridge-range`. For new measurements use `python scripts/reproduce.py run ridge-range`. See [reproduction details](../../docs/reproducing.md) for data, source versions, and limitations.
