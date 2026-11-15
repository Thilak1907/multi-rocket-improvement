# Reports

| Report | Scope | Rebuild command |
| --- | --- | --- |
| [Corrected review](review_revision/report.md) | Current recommendation, ablation, resource use, accuracy interval | `python scripts/reproduce.py summarize review` |
| [Paper comparison](paper_comparison/comparison.md) | Released author results and project accuracy | `python scripts/compare_paper.py` |
| [Additional-view ensemble](accuracy_search/report.md) | Optional accuracy experiment | `python scripts/reproduce.py summarize accuracy` |
| [Feature maps](feature_power_search/report.md) | Negative square-root/log results | `python scripts/reproduce.py summarize feature-power` |
| [Ridge range](ridge_range_search/report.md) | Negative wider-alpha result | `python scripts/reproduce.py summarize ridge-range` |
| [Legacy combination search](combination_search/report.md) | Earlier first-difference alignment | `python scripts/reproduce.py summarize legacy-combinations` |
| [Early optimization comparison](optimization_comparison/report.md) | Initial pooling/ridge/memory study | `python scripts/reproduce.py summarize legacy-optimization` |
| [Initial timing study](optimization_comparison_v1/report.md) | Historical three-repeat context | `python scripts/reproduce.py summarize legacy-initial` |

Supporting CSVs contain per-dataset metrics and summaries. Detailed raw jobs are
in `reproducibility/records.zip`; rebuilding analysis writes expanded records
under ignored `output/`. Historical and corrected feature definitions must not
be pooled. See [reproduction details](../docs/reproducing.md) for training commands
and the two early studies' source-fidelity limitation.
