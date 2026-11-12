# Benchmark reports

- [Current corrected comparison](review_revision/report.md): recommended 30k bundle, size-routing decision, three accuracy resamples, ablation, and trade-off chart.
- [Previous combination search](combination_search/report.md): legacy first-difference alignment; retained for comparison with the earlier recommendation.
- [Optimization comparison](optimization_comparison/report.md): original optimization and paper/repository comparisons, using legacy alignment.
- [Initial optimization run](optimization_comparison_v1/report.md): historical timing context referenced by the optimization report.

CSV files alongside each report contain the supporting measurements. Historical
and corrected results use different features and should not be pooled.
Raw jobs, source snapshots, and local datasets are excluded from Git under
`output/` and `data/`; retain those directories locally for detailed replay.
