# Paper and project comparison

Published accuracy is calculated from the authors' released CSVs, not transcribed as a new table from the article. The 55-name subset is the project confirmation set. Kernel seeds, feature correctness mode, and number of resamples differ; these rows are context, not a paired superiority test.

| Source | Datasets | Resamples | Features | Mean accuracy |
| --- | ---: | --- | ---: | ---: |
| Authors released 50k results | 109 | 30 | 49,728 | 87.967% |
| Authors released 50k results, project subset | 55 | 30 | 49,728 | 87.654% |
| Authors released 50k results, project subset | 55 | 1 (index 1) | 49,728 | 87.987% |
| Project corrected 50k | 55 | 3 (indices 1–3) | 49,728 | 87.399% |
| Project corrected 30k | 55 | 3 (indices 1–3) | 29,568 | 87.285% |

The paper also reports 15.77 minutes of 50k training over 112 problems in its single-thread timing table (Table 2). Our primary timing aggregate covers 55 datasets with four Numba threads and one BLAS thread on different hardware, so no speed-up against that paper timing is calculated. The project's 1.66× figure uses its own paired corrected 50k baseline.

Sources: [paper](https://doi.org/10.1007/s10618-022-00844-1), [authors' released results](https://github.com/ChangWeiTan/MultiRocket/tree/3ccaa4f8ed11769b904af885e17018c39c89a2ac/results).
