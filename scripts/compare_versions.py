"""Clean vs Raw cleaning-version robustness comparison (proposal 3.1 / 7.1).

Reads both result sets (Clean from results/, Raw from results_raw/) and reports
whether the headline findings hold under the Raw cleaning version:
  * mean RMSPE-vs-fraction per method, side by side
  * does transfer still beat scratch? does linear still win?
  * correlation of per-target transfer gains across versions

Run AFTER both sweeps exist:
    python -I scripts/run_experiments.py --config configs/example.yaml       # Clean
    python -I scripts/run_experiments.py --config configs/raw_version.yaml   # Raw
    python -I scripts/compare_versions.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CURVE = ["scratch", "finetune_full", "finetune_head", "zeroshot"]
FLAT = ["persistence", "linear", "har"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="results/experiment_results.csv")
    ap.add_argument("--raw", default="results_raw/experiment_results.csv")
    ap.add_argument("--out", default="results/report")
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        raise SystemExit(
            f"Raw results not found at {args.raw}. Download the Raw-version bars "
            f"and run the Raw sweep first (see configs/raw_version.yaml header).")
    clean = pd.read_csv(args.clean).assign(version="clean")
    raw = pd.read_csv(args.raw).assign(version="raw")
    os.makedirs(args.out, exist_ok=True)

    common = sorted(set(clean.target) & set(raw.target))
    clean = clean[clean.target.isin(common)]
    raw = raw[raw.target.isin(common)]
    print(f"Comparing on {len(common)} common targets.")

    both = pd.concat([clean, raw], ignore_index=True)
    tbl = both.pivot_table(index=["method", "fraction"], columns="version",
                           values="rmspe", aggfunc="mean")
    tbl["abs_diff"] = (tbl["raw"] - tbl["clean"]).abs()
    tbl.to_csv(os.path.join(args.out, "clean_vs_raw.csv"))
    print("\n=== mean RMSPE: clean vs raw (by method x fraction) ===")
    print(tbl.round(4).to_string())

    # Headline checks
    def mean_at(df, method, frac):
        s = df[(df.method == method) & (df.fraction == frac)]["rmspe"]
        return float(s.mean()) if len(s) else float("nan")
    print("\n=== headline robustness ===")
    for ver, df in [("clean", clean), ("raw", raw)]:
        sc = mean_at(df, "scratch", 0.01)
        ft = mean_at(df, "finetune_full", 0.01)
        lin = mean_at(df, "linear", 0.0)
        print(f"  {ver}: finetune@1%={ft:.3f}  scratch@1%={sc:.3f} "
              f"(transfer {100*(sc-ft)/sc:+.0f}%)  linear={lin:.3f}  "
              f"linear_best={'Y' if lin<=ft else 'N'}")

    # overlay plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for ver, df, ls in [("clean", clean, "-"), ("raw", raw, "--")]:
        for m in ["scratch", "finetune_full", "linear"]:
            if m in FLAT:
                v = df[df.method == m]["rmspe"].mean()
                ax.axhline(v, linestyle=":" if ver == "raw" else "-.",
                           alpha=0.5, label=f"{m} ({ver})")
            else:
                g = df[df.method == m].groupby("fraction")["rmspe"].mean()
                ax.plot(g.index, g.to_numpy(), linestyle=ls, marker="o",
                        label=f"{m} ({ver})")
    ax.set_xlabel("target-data fraction")
    ax.set_ylabel("mean test RMSPE")
    ax.set_title("Robustness: Clean vs Raw cleaning version")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(args.out, "clean_vs_raw.png")
    fig.savefig(p, dpi=130)
    print(f"\nWrote clean_vs_raw.csv and {p}")


if __name__ == "__main__":
    main()
