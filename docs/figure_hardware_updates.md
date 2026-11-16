# Figure and environment revisions

The updated plot is available as
[PDF](../reports/review_revision/tradeoffs_all.pdf),
[PNG](../reports/review_revision/tradeoffs_all.png), and
[SVG](../reports/review_revision/tradeoffs_all.svg).
It has 10-point axes, direct labels instead of a large legend, and two zoom
panels with unchanged data coordinates. The retained `fast30` model is the
orange star. All twelve configurations remain present.

A local manuscript copy with these edits is at
`output/manuscript_audit/manuscript.figure-hardware.tex`; the figure files are
alongside it. It starts from the previously checked source and preserves its
result tables. The supplied attachment is unchanged.

## 1. Replace the existing trade-off figure

In “Efficiency and accuracy on resample 1”, replace the whole block from
`\begin{figure}[t]` through `\end{figure}` for `fig:tradeoff` with:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{tradeoffs_all.pdf}
\caption{Accuracy and speed on 55 datasets, corrected MultiRocket features,
resample 1 and one common timing run. Panels (b) and (c) zoom nearby points
without jitter. The orange star is the retained 30k blocked-ridge model
(\texttt{fast30}); 30k routed and sklearn denote \texttt{auto30} and
\texttt{fused30}. Budgets are nominal. Optimized four-operator models use
buffered extraction, in-place processing, 256-row batches and folded scaling;
50k buffers omits folding, 50k sklearn bundle includes it, and 50k blocked
uses the blocked head. The 50k reference omits that optimization bundle.
HYDRA-only uses 64 groups; the hybrid combines 20k MultiRocket with 32 HYDRA
groups; the LSPV+MPV pair retains 24,864 columns. The dotted horizontal line
is reference accuracy, the solid vertical line is $1\times$ speed-up, and
the dashed line joins the observed accuracy/speed Pareto frontier, which
does not account for memory. The main table uses three timing repetitions.}
\label{fig:tradeoff}
\end{figure*}
```

Put `tradeoffs_all.pdf` next to the `.tex` file or adapt its graphics path.
Use the full two-column `\textwidth`; reducing it to one column defeats the
font-size correction. Zoom panels deliberately magnify small differences and
are not evidence that the tiny accuracy differences are significant.

Immediately before this figure, after “the two should be read separately”, add:

```latex
Two additional representation controls are standalone HYDRA with 64 groups
(\texttt{hydra64}) and a nominal 50k kernel-budget model retaining only LSPV
and MPV (\texttt{pair50}, 24,864 columns); neither is the retained configuration.
```

The [identifier/label mapping](../reports/review_revision/plot_labels.csv) covers
all points. The [plot metadata](../reports/review_revision/plot_metadata.json)
records their original accuracy and speed coordinates.

## 2. Main machine: “Code and environment”

Replace the generic “one macOS ARM64 machine” description with:

```latex
The main benchmark ran on an Apple M5 machine (Mac17,3), with 10 physical
cores, 10 logical CPUs and 24\,GiB installed RAM (25.77\,GB), running macOS
26.5.2. It used four Numba and Torch threads and one BLAS thread, with
Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1,
Numba 0.67.0 and Torch 2.13.0.
```

The CPU/model/RAM values were read using `sysctl` during this revision on
16 November 2026. Earlier saved CLI CSVs corroborate 10 physical cores,
10 logical CPUs, and RAM rounded to 26 decimal GB. The original protocol did
not record the chip brand; the hardware verification provenance is retained in
[`reproducibility/environment.json`](../reproducibility/environment.json).

## 3. Give Section IV its own environment paragraph

After “Code and environment” and before “Seeding”, add:

```latex
\textbf{Second-machine reproduction environment.} The Section IV checks
used a Linux VM reporting an Intel Xeon at 2.30\,GHz, with two allocated
vCPUs and an 8\,GB cgroup memory limit. This is the workload limit, not
the host's installed RAM. Numba and OpenMP used two threads; OpenBLAS
and MKL used one. The stack was Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1, Numba 0.68.1, pandas 3.0.5 and CPU-only Torch.
The exact Torch version was not recorded. This setup is separate from
the main macOS benchmark.
```

Source: [the recorded Linux environment](../reproduction/recorded/environment.txt).
“8 GB” is quoted from that record; an exact byte count for the cgroup limit
was not retained. Do not replace it with an invented host RAM capacity or a
more specific Xeon model.

## 4. Reconcile Crop in the introduction and Section IV

Replace the introduction's ElectricDevices/Crop sentence with:

```latex
In the main macOS benchmark, a corrected 50k model exceeded the
9\,GiB process-memory guard on ElectricDevices. In the separate Linux
checks with an 8\,GB cgroup limit, both the legacy and corrected 50k
processes were killed on Crop; the recorded kernel logs corroborate
out-of-memory kills.
```

In Section IV's “Other lessons”, use:

```latex
Under the Linux 8\,GB cgroup limit the legacy 50k model peaked at
6.4\,GB (5.95\,GiB) on StarLightCurves, and both legacy and corrected
50k processes were OOM-killed on Crop. These kills are distinct from
the supervisor's 9\,GiB guard stops in the macOS benchmark.
```

The recorded JSONL has separate failed Crop attempts for `paper50` and
`fixed50`. These mean legacy and corrected 50k, respectively. The
[kernel log](../reproduction/recorded/crop_oom_kernel_log.txt) contains two
memory-cgroup OOM entries. The checked-in reproduction table also lists both.

## 5. Acknowledge the published/reference accuracy gap

In the evaluation “Setup”, before “Efficiency and accuracy on resample 1”, add:

```latex
For context, at the common archive index 1 on the same 55 names, the
authors' released 50k results average 87.987\% versus 87.760\% for our
corrected reference, a 0.227-point gap; kernel seeds and feature alignment
differ, while the broader accuracy analyses also differ in coverage
(30 released resamples versus our three).
```

Both numbers in the 0.227-point comparison use index 1. Resample coverage
therefore must not be presented as the cause of that particular gap. It is a
limitation of the broader three-versus-30 comparison. No single cause of the
gap has been experimentally isolated.

## Reproduction

```bash
python -m experiments.plot_tradeoffs
```

This renders the checked-in first-run table. `summarize_revision.py` now calls
the same renderer. The checked source bundle also includes the updated analysis
and renderer, so a fresh `scripts/reproduce.py summarize review` produces the
same presentation without changing the pinned numerical runtime sources.
Existing workspaces created with the old bundle retain their hash protection;
use a fresh `--workspace` directory after a source-bundle update.

The new manuscript copy passes the numeric-table audit. A PDF manuscript build
still requires a TeX installation and the user's `refs.bib`, neither of which
was supplied for this check.
