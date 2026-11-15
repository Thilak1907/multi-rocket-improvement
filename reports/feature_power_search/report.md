# Fixed nonlinear feature maps for the retained 30k bundle

The production default remains unchanged by this experiment. Every model keeps the corrected 30k bundle, all four pooling operators, blocked ridge, buffered extraction, in-place processing, 256-row prediction batches, and folded scaling. Each candidate has 29,568 columns and one ridge head, without an ensemble.

The sole change is a fixed pointwise feature map before StandardScaler: signed square root `sign(z)*sqrt(abs(z))`, or signed logarithm `sign(z)*log1p(abs(z))`. Maps are applied in place, with the same kernels and seeds as baseline. There are no fitted map parameters or test-distribution statistics. These maps are a hypothesis, not sklearn's fitted Yeo-Johnson/Box-Cox PowerTransformer.

Background: nonlinear feature transforms can change skewness and scale effects ([sklearn preprocessing documentation](https://scikit-learn.org/stable/modules/preprocessing.html#power-transforms)); this does not guarantee improved classification.

## Frozen selection

54 selection dataset names; stratified TRAIN-only 80/20 partitions, capped at 512 cases in each partition. Choose the largest mean validation gain, provided it is at least +0.1 percentage points and summed fit-plus-predict runtime is at most 1.20 times baseline. Candidates are not selected using confirmation labels.

| Map | Validation accuracy | Change | Runtime / baseline |
| --- | ---: | ---: | ---: |
| sqrt | 84.935% | -0.420 pp | 1.009× |
| log | 85.246% | -0.109 pp | 1.026× |

**Neither map passed selection. Retain the current recommendation.**

## Limits and reproduction

One timing measurement per split; small runtime differences are descriptive. Feature-map processing is included in fit/predict time. Baseline is shared across comparisons and evaluated first; candidate order is rotated. Process high-water RSS includes imports/warmup and earlier fits, so it is an upper bound rather than isolated candidate memory. Every dataset worker has a 9 GiB polling guard.

Confirmation names were inspected during earlier development; these are follow-up paired measurements, not untouched external validation. Resamples reuse examples. This is a finite benchmark comparison, not a universal accuracy guarantee. No hardware, energy, multivariate, or GPU improvement is established.

Rebuild the recorded tables with `python scripts/reproduce.py summarize feature-power`. For new measurements use `python scripts/reproduce.py run feature-power`. See [reproduction details](../../docs/reproducing.md) for data, source versions, and limitations.
