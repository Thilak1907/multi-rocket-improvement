"""Summarize the Section IV reproduction checks into a Markdown report.

Compares the runs in a results JSONL file with the authors' published accuracies:
per-fold results from results/resamples/MULTIROCKET_{50K,10K}_TESTFOLDS.csv
(installed by `python scripts/download_data.py --upstream`) and the resample-0
values in reproduction/recorded/published_resample0.csv.

Usage:
    python reproduction/analyze.py                                    # recorded results
    python reproduction/analyze.py --results reproduction/output/results.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def table(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def folds(path):
    f = pd.read_csv(path).iloc[:, 1:].set_index("folds:")
    f.columns = f.columns.astype(int)
    return f


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=HERE / "recorded/results.jsonl")
    parser.add_argument("--upstream-results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=HERE / "output/report.md")
    args = parser.parse_args()

    recs = [json.loads(line) for line in args.results.open()]
    runs = pd.DataFrame([r for r in recs if "acc" in r])
    rss = pd.DataFrame([r for r in recs if "peak_rss_mib" in r])
    failed = pd.DataFrame([r for r in recs if r.get("status") in ("killed", "error")])

    pub_r0 = pd.read_csv(HERE / "recorded/published_resample0.csv").set_index("dataset")["accuracy"]
    f50 = folds(args.upstream_results / "resamples/MULTIROCKET_50K_TESTFOLDS.csv")
    f10 = folds(args.upstream_results / "resamples/MULTIROCKET_10K_TESTFOLDS.csv")
    md = ["# Section IV reproduction checks", "", f"Source: `{args.results}`", ""]

    # Stage A: seed variance on the original split
    a = runs[(runs.stage == "A") & (runs.config == "paper50")]
    if len(a):
        rows = []
        for d, g in a.groupby("dataset"):
            acc = g.acc.to_numpy()
            rows.append(dict(dataset=d, paper=100 * pub_r0[d], ours_mean=100 * acc.mean(),
                             ours_sd=100 * acc.std(ddof=1), seed_min=100 * acc.min(), seed_max=100 * acc.max(),
                             in_range="yes" if acc.min() - 1e-9 <= pub_r0[d] <= acc.max() + 1e-9 else "NO"))
        md += ["## A. Original split, paper configuration, 10 kernel seeds (accuracy %)", "",
               table(pd.DataFrame(rows), 2), ""]

    # Stage B: paired folds against the published 50k folds
    b = runs[runs.stage == "B"]
    if len(b):
        rows = []
        for d, g in b[b.config == "paper50"].groupby("dataset"):
            g = g.sort_values("resample")
            ours = g.acc.to_numpy()
            pub = f50.loc[d, g["resample"].to_numpy()].to_numpy(float)
            diff = ours - pub
            p = wilcoxon(ours, pub).pvalue if np.any(diff != 0) else np.nan
            rows.append(dict(dataset=d, folds=len(g), ours=100 * ours.mean(), paper=100 * pub.mean(),
                             diff_pp=100 * diff.mean(), lower=int((diff < 0).sum()), equal=int((diff == 0).sum()),
                             higher=int((diff > 0).sum()), wilcoxon_p=p))
        tb = pd.DataFrame(rows)
        md += ["## B1. Thirty resamples, seed 0, paired with the published 50k folds (accuracy %)", "",
               table(tb, 3), "", f"Mean over datasets: ours {tb.ours.mean():.2f}, paper {tb.paper.mean():.2f}.", ""]

        rows = []
        for d, g in b[b.config == "fast10"].groupby("dataset"):
            g = g.sort_values("resample")
            rows.append(dict(dataset=d, ours_10k=100 * g.acc.mean(),
                             paper_10k=100 * f10.loc[d, g["resample"].to_numpy()].astype(float).mean()))
        if rows:
            md += ["## B2. 10k configuration against the published 10k folds (accuracy %)", "",
                   table(pd.DataFrame(rows), 2), ""]

        piv = b.pivot_table(index=["dataset", "resample"], columns="config", values="acc")
        means = piv.groupby("dataset").mean()
        rng = np.random.default_rng(0)
        tot = b.assign(t=b.fit_s + b.predict_s).pivot_table(index="dataset", columns="config", values="t", aggfunc="sum")
        rows = []
        for c in [c for c in ["paper50", "fixed50", "rec30", "fast10"] if c in means]:
            dd = (means[c] - means["paper50"]) * 100
            boots = [rng.choice(dd.to_numpy(), len(dd)).mean() for _ in range(10000)]
            rows.append(dict(config=c, accuracy=100 * means[c].mean(), delta_pp=dd.mean(),
                             ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                             time_s=tot[c].sum(), speedup=tot["paper50"].sum() / tot[c].sum()))
        md += ["## B3. Configurations paired with the paper configuration (8 datasets x 30 resamples)", "",
               "Delta is the mean of per-dataset differences; the CI is a dataset bootstrap (10,000 draws, seed 0).", "",
               table(pd.DataFrame(rows), 2), ""]

        if "fixed50" in piv:
            rows = []
            for d in means.index:
                x, y = piv.loc[d, "fixed50"].to_numpy(), piv.loc[d, "paper50"].to_numpy()
                rows.append(dict(dataset=d, diff_pp=100 * (x - y).mean(),
                                 wilcoxon_p=wilcoxon(x, y).pvalue if np.any(x != y) else np.nan))
            md += ["## B4. Effect of the first-difference fix (fixed50 minus paper50, 30 resamples)", "",
                   table(pd.DataFrame(rows), 3), "", "Empty p-value: identical accuracy on every fold.", ""]

        p = b[b.config == "paper50"]
        total = (p.fit_s + p.predict_s).sum()
        md += ["## B5. Where the time goes (paper configuration, all stage-B runs)", "",
               f"- Prediction (almost all of it the test-set transform): {100 * p.predict_s.sum() / total:.1f}% of fit+predict time",
               f"- Classifier fitting: {100 * p.classifier_fit_s.sum() / total:.1f}%", ""]

    # Stage C: large datasets
    c = runs[runs.stage == "C"]
    if len(c) or len(failed):
        rows = []
        for (d, cfg), g in c.groupby(["dataset", "config"]):
            peak = rss[(rss.dataset == d) & (rss.config == cfg)].peak_rss_mib
            rows.append(dict(dataset=d, config=cfg, accuracy=100 * g.acc.iloc[0], paper_r0=100 * pub_r0.get(d, np.nan),
                             time_s=(g.fit_s + g.predict_s).iloc[0],
                             peak_rss_gib=peak.iloc[0] / 1024 if len(peak) else np.nan, status="ok"))
        for r in failed.itertuples():
            if r.stage == "C":
                rows.append(dict(dataset=r.dataset, config=r.config, accuracy=np.nan,
                                 paper_r0=100 * pub_r0.get(r.dataset, np.nan), time_s=np.nan, peak_rss_gib=np.nan,
                                 status=f"{r.status} (exit {r.exit_code})"))
        md += ["## C. Large datasets, original split, seed 0", "",
               "Exit -9 is SIGKILL; on the recorded 8 GB machine the kernel log showed out-of-memory kills.", "",
               table(pd.DataFrame(rows).sort_values(["dataset", "config"]), 2), ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
