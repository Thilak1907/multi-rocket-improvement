# Environment and comparison revisions

The previous trade-off image and plotting code have been restored. The hardware,
Crop clarification, and released/reference accuracy-gap updates below remain.
The local source copy is `output/manuscript_audit/manuscript.figure-hardware.tex`;
its original figure block has also been restored.

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

## Figure reproduction

Use `python scripts/reproduce.py summarize review` to rebuild the original
figure from the recorded runs. The pinned source bundle has also been restored.
