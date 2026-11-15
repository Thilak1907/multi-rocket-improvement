# Implementation notes

This fork evaluates resource reductions in the univariate MultiRocket pipeline.
The underlying convolution patterns, raw/difference representations, and four
pooling operators come from [MultiRocket](https://doi.org/10.1007/s10618-022-00844-1).
The experiments concern implementation cost and feature-budget trade-offs;
they do not establish a new accuracy-leading classifier.

## Pipeline

For a training matrix with shape `(n_series, series_length)`, the model:

1. Forms the original series and its first difference, `x[t+1] - x[t]`.
2. Fits separate dilations and response-quantile thresholds for both views.
3. Applies 84 length-nine kernel patterns: three weights of +2 and six of −1.
4. Pools each response using PPV, LSPV, MPV, and MIPV.
5. Cleans and standardizes feature columns using training statistics.
6. Fits ridge classification with leave-one-out mean squared error to select alpha.

The nominal feature count rounds down in groups of 672. Nominal 50k produces
49,728 columns; nominal 30k produces 29,568. Reducing the budget generates fewer
features from the start. It is not learned feature or kernel pruning.

## Retained changes

| Change | Implementation | Effect |
| --- | --- | --- |
| 30k feature budget | `multirocket.py` | Smaller representation; predictions can change |
| Blocked ridge | `fast_ridge.py` | Build the sample-space Gram matrix in column blocks |
| Buffered extraction | `corrected.py` | Reuse per-example scratch arrays |
| In-place processing | `multirocket.py` | Avoid cleaning/scaling copies of the feature matrix |
| Prediction batches | `multirocket.py` | Extract at most 256 test rows at once |
| Folded scaling | `multirocket.py` | Absorb feature scaling into the linear decision coefficients |
| Difference-length correction | `corrected.py` | Use length L−1 in the difference convolution branch |

The `previous30` preset selects this corrected bundle. The base constructor
retains its original 50k/sklearn defaults. `candidate30` enables size routing;
`legacy30` restores the earlier first-difference behavior.

### Ridge computation

For centered features A, the wide-data solver accumulates
`K = A @ A.T = sum(block @ block.T)` in float64, then reuses an eigendecomposition
for the alpha grid. Coefficients are recovered only for the selected alpha.
Leave-one-out residual denominators include the unpenalized-intercept correction.

The original alpha grid has ten values from 10^-3 to 10^3. Tall matrices and
near-tied LOO scores fall back to sklearn. The implementation targets dense,
unweighted classification with the default squared-error criterion, not every
mode of RidgeClassifierCV. The optional `ridge_alphas` argument changes the grid;
`None` preserves the original values.

Blocking bounds float64 feature-conversion workspace, but the Gram matrix still
needs O(n²) memory and its eigendecomposition O(n³) time. Training still constructs
a complete float32 training-feature matrix. This is not streaming training.

### Inference and memory

A float32 feature matrix uses approximately `4 * n * F` bytes. Prediction batching
bounds that buffer by `4 * min(256, n_test) * F`; at 30k this is 28.875 MiB for a
full batch. Total process RSS also includes inputs, scratch buffers, libraries,
coefficients, and solver workspace.

For a score `W @ ((z - mean) / scale) + intercept`, folded prediction stores
`W_folded = W / scale` and `b_folded = intercept - W_folded @ mean`.
This avoids a separate scaling pass, but keeps another coefficient copy.
Mathematical equivalence does not guarantee identical rounding; tests and paired
prediction checks cover the measured cases.

Training features were already local to the inherited `fit` method. Explicit
deletion clarifies lifetime; the material savings come from avoiding copies and
bounding test-feature allocation.

### Feature semantics and correctness

The difference series has length L−1. The legacy difference branch used L in
one accumulation endpoint; corrected extraction uses L−1. Both the fresh 50k
baseline and the retained 30k model use the correction. Historical results use
the legacy behavior and remain separate.

Other inherited pooling conventions are preserved. In particular, MPV sums
`response + bias` after testing `response > bias`, and LSPV uses the original
endpoint/equality rules. Replacing these with an intuitive implementation of
the operator names can change features. Independent convolution tests check
raw/difference responses and all operator subsets.

## Alternatives evaluated

- Removing two operators halves columns at a fixed kernel budget. It was tested
  but not retained as the recommended representation.
- Size routing uses sklearn below 693 MiB of estimated training features and
  blocked ridge above it. It did not beat the fixed backend under the adoption rule.
- The HYDRA hybrid concatenates separately scaled feature blocks and fits one
  shared ridge classifier. Upstream code and licensing are retained in `experiments/vendor`.
- A coarse-view ensemble averages scores from two 30k models. Its observed
  accuracy gain was inconclusive and its runtime was 1.62× the single model.
- Signed square-root/log feature maps and a wider alpha grid failed TRAIN-only
  selection. Their negative results are retained.

None of these optional experiments changes the recommended preset.

## Scope of the evidence

The corrected timing comparison uses 55 dataset names, resample 1, and three
timing repetitions. The main accuracy analysis uses resamples 1–3, first averaging
paired changes within each dataset. Datasets, rather than timing repetitions or
165 independent rows, are the statistical units.

The −0.3-point non-inferiority margin was fixed before the corrected runs. The
retained bundle's paired mean change is −0.114 points, with dataset-bootstrap
95% CI [−0.261, +0.041]. This supports the stated tolerance, not equality on every
dataset. Balanced accuracy and macro F1 do not improve in every comparison.

Confirmation names had been inspected during prior development. Three resamples
are fewer than the paper's 30, and reused observations and related datasets
limit independence. Timings exclude imports/loading/JIT warmup; RSS includes
them. Resource-guard stops are not documented OS OOM events. No multivariate,
GPU, or energy improvement is established by these measurements.

## Other repository paths

`main_ucr.py` and `main_ucr_109.py` expose the revised univariate model.
`main_mtsc.py` and `multirocket_multivariate.py` are inherited multivariate code;
the optimization claims do not extend to that path. The optional PyTorch
`logistic_regression.py` is a linear head over extracted features, not a learned
convolutional network. It is not used in the reported ridge comparisons.

See [reproduction instructions](reproducing.md) for commands and artifact details,
[the corrected report](../reports/review_revision/report.md) for measurements,
and [the report index](../reports/README.md) for all experiments.
