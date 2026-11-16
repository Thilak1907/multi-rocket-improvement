# Manuscript and software check

Checked 16 November 2026 against the supplied IEEEtran source. Source SHA-256:
`c438dcbd9a8cd70a35e3961d8990d759830737afec79a67e26a73aea7b978a60`.

## Outcome

The four main result tables match the retained measurements: 52 cells checked,
including 51 numeric values and the baseline memory dash. The numerical
non-inferiority result, job counts, seed summary, memory conversion, and parity
counts also agree. The source still has scientific wording that needs correction;
matching the tables does not validate those statements.

An edited source copy was prepared locally at
`output/manuscript_audit/manuscript.checked.tex`. Its result tables are unchanged.
The original copy, edit list, table comparisons, fresh subset results, and figure
are alongside it. The attachment itself was not modified. The copy is a suggested
revision for author review, not a compiled or publication-ready PDF.

## Software checks

| Check | Outcome |
| --- | --- |
| Unit tests, pinned Numba 0.67.0 / llvmlite 0.49.0 / NumPy 2.4.6 | 33 passed |
| Unit tests, isolated Numba 0.68.1 / llvmlite 0.50.0 / NumPy 2.4.6 | 33 passed |
| Dataset and resample-index fingerprints | 218 TSVs and 109 TRAIN-index files verified |
| Reanalysis of all seven recorded studies | Saved numeric summary tables reproduced |
| Recommended-model ArrowHead CLI smoke | Passed |
| Three-dataset Stage B rerun | 360 successful fits; analysis completed |
| CLI help for the three original entry points | Passed in the installed environment |
| Logistic head CPU smoke, string labels, one/two/three classes | Passed |
| Internal LaTeX section/table/figure labels | All referenced labels present |
| LaTeX compilation | Not checked: no TeX engine or `refs.bib` was supplied |

The alternate-compiler test does not reproduce the reported Linux floating-point
assertion failure. It used Python 3.11 on this machine, not the recorded Linux
Python 3.13 stack. No assertion was weakened to make the test suite pass.
The multivariate help check is not a multivariate training benchmark.

### Fresh results are not universally bitwise identical

The new Stage B run used Coffee, Beef, and ECG200; four configurations; and
resamples 0–29 at seed 0. Of 360 accuracy values, 358 matched the old Linux
records exactly. Two differed:

| Configuration | Dataset | Resample | New accuracy | Recorded accuracy |
| --- | --- | ---: | ---: | ---: |
| fixed50 | Beef | 23 | 70.000% | 73.333% |
| fast10 | Beef | 16 | 70.000% | 73.333% |

Each difference is one fewer correct case out of 30 test cases; the number of
changed prediction labels cannot be inferred from accuracy alone. Coffee and
ECG200 matched in every run, and all 90 retained-configuration (`rec30`) values
matched. These checks use a different hardware/library stack from the original
Section IV records. The cause of the two discrepancies was not isolated; neither
compiler version nor a single other dependency is established as the cause.
The original measurements were not overwritten to hide the differences.

## Numerical claims verified

| Quantity | Recomputed value | Manuscript rounding |
| --- | ---: | ---: |
| Main fixed-30k aggregate speed-up | 1.65742× | 1.66× |
| Main median paired RSS reduction | 30.9790% | 31.0% |
| Three-resample mean accuracy change | −0.113668 pp | −0.114 pp |
| Dataset-bootstrap interval | [−0.261081, +0.041069] pp | [−0.261, +0.041] pp |
| One-sided non-inferiority p-value | 0.0102330 | 0.01023 |
| Corrected review job records | 2,021 | 2,021 |
| Alpha/prediction parity comparisons | 275 | 275 |
| Ensemble baseline hash matches | 165 | 165 |
| Standalone reproduction JSONL records | 1,358 | 1,358 |
| ArrowHead sample standard deviation | 0.648739 pp | 0.65 pp |
| ArrowHead seed range | 85.7143%–88.0000% | 85.71%–88.00% |
| Eight-dataset mean accuracy effect of the fix | −0.012407 pp | −0.01 pp |
| Smallest nonidentical per-dataset fix Wilcoxon p | 0.179712 | greater than 0.17 |
| Prediction share of fit-plus-predict time | 70.4731% | about 70% |
| Classifier-fit share | 9.92778% | about 10% |
| StarLightCurves `paper50` peak | 6.384484 GB = 5.946014 GiB | 6.4 GB = 5.95 GiB |

See [table checks](../reports/manuscript_audit/table_checks.csv) and
[numeric evidence](../reports/manuscript_audit/numeric_summary.json). All finalist,
stage-time, three-resample metric, and ensemble table entries passed their
printed-precision checks.

## Corrections needed in the supplied source

Line numbers below refer to the original attachment, before edits.

| Lines | Issue | Correction in the local checked copy |
| --- | --- | --- |
| 46–53 | Abstract can imply full-archive reproduction and correction of every legacy comparison | Limit reproduction to the checked eight datasets; corrections apply to the optimization comparison; identify three accuracy resamples |
| 126–127 | Padding is said to alternate between bias indices | It alternates with kernel and dilation indices; thresholds share that response |
| 130–134 | LSPV is simplified to a generic run counter; MPV's offset is claimed to vanish under scaling | Preserve the implemented boundary convention; acknowledge the empty-response exception below |
| 161–166 | Main and Section IV environments can be conflated; 20261111 is called every model's seed | Distinguish the Linux stack and call 20261111 the base seed for dataset-specific hashes |
| 197 | “Within 0.11 pp” treats a rounded value as an exact bound | Use “within about 0.11 pp”; Beef's mean change is approximately 0.111111 pp |
| 199–200 | Normalization and author-file disagreement are said to be ruled out | State which reference is used and leave remaining causes unresolved |
| 206–214 | Exact oracle tolerance, maximum difference, and 87% affected-features estimate lack a saved supporting experiment | Replace with the actual scope of the independent-oracle tests; do not treat the original figures as disproven, but require an artifact before retaining them |
| 211–214, 598 | “Accuracy unchanged” and a causal explanation for the error going unnoticed are too strong | Describe little observed mean change, not equality; use a qualified explanation |
| 249–250 | Pooling is said to be exactly the paper definition despite the implementation-specific MPV convention | Say the inherited conventions are retained, with the stated corrections |
| 270–271 | The full float64 feature copy is said never to occur | Restrict the claim to blocked accumulation; sklearn fallbacks can allocate workspace |
| 594–598 | Conclusion extends the small reproduction check too broadly | State the eight-dataset scope explicitly |

### Why the MPV explanation needs changing

The code returns `mean(c + b)` when there is an exceedance, and zero otherwise.
The corresponding `mean(c - b)` is lower by `2*b` only for nonempty responses.
If one series has no exceedances, both feature values are zero, so the difference
is not a constant across that feature column.

For example, with `b=2`, three series can produce released MPVs `[0, 6, 8]`
and alternative MPVs `[0, 2, 4]`. Standardizing the two columns does not make
them identical. The manuscript's general claim that standardization removes
the discrepancy is therefore false. This does not change the measured project
results: all compared models use the same preserved MPV convention.

The local implementation behavior is visible in `multirocket/pooling.py`;
the original paper describes pooling in Section 3.4:
[MultiRocket](https://doi.org/10.1007/s10618-022-00844-1).

### Limits of the oracle claims

`tests/test_first_difference.py` verifies explicit convolution on integer-valued
inputs over several lengths, dilations, all operators, and operator subsets.
It separately checks float-input buffered/full parity. It does not save an
experiment establishing a universal 10^-15 float-feature tolerance, a maximum
legacy difference of 20, or an 87% affected-feature rate. Those quantities
depend on the inputs and definition being compared. The checked copy replaces
unbacked precision with the available test evidence.

### Other claims checked

The six logistic-head repairs are consistent with the diff against the pinned
upstream source: raw logits for the loss functions, floating binary targets,
correct validation-step normalization, a model initialized for return, and
label encoding. The CPU smoke verifies the repaired path runs; it is not an
accuracy comparison or evidence for multivariate/GPU performance.

The provided author result files differ for several resample-0 entries. For
example, InsectWingbeatSound is 67.273% in the single-split file versus 67.475%
in the per-fold file; OSULeaf is 95.455% versus 96.694%. Using the corresponding
file consistently is necessary, but does not by itself rule out every source
of disagreement with the paper.

## Repeating the checks

```bash
python -m unittest discover -s tests -v
python scripts/reproduce.py verify
python scripts/reproduce.py summarize all
python scripts/reproduce.py run review --smoke
python reproduction/analyze.py
python scripts/audit_manuscript.py /path/to/manuscript.tex
```

For the three-dataset rerun, use a new output file (the runner appends):

```bash
python reproduction/run_all.py B --datasets Coffee,Beef,ECG200 \
  --configs paper50,fixed50,rec30,fast10 \
  --out output/manuscript_audit/new_subset/results.jsonl
python reproduction/analyze.py \
  --results output/manuscript_audit/new_subset/results.jsonl \
  --out output/manuscript_audit/new_subset/report.md
```

The full 55-dataset benchmark was reanalysed from its records, not retrained in
this audit. The 8 GB Crop out-of-memory attempts were not deliberately repeated.
Fresh timings are not compared directly to the manuscript's old-machine timings.
The supplied `refs.bib` is also needed before a complete LaTeX build can be
verified; the figure was copied alongside the suggested local source.
