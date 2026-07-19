"""
Histogram of per-label PR-AUC (best config), with the two confidence-tier
median thresholds marked:
  Scenario A - median computed across all 138 fine labels
  Scenario B - median computed within the high-n_PA subset only
               (n_PA >= median n_PA, i.e. labels where PA is reliable)

Usage:
    python plot_prauc_threshold_scenarios.py

Inputs:
    pa_vs_prauc_merged.csv   (produced by plot_pa_vs_prauc_scatter.py)

Outputs:
    prauc_threshold_scenarios.png
"""

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- CONFIG -------------------------
MERGED_CSV = "pa_vs_prauc_merged.csv"
OUTPUT_PNG = "prauc_threshold_scenarios.png"
# ------------------------------------------------------------


def main():
    df = pd.read_csv(MERGED_CSV)

    # Scenario A: median over all labels regardless of PA validity
    median_all = df["PR_AUC_mean"].median()

    # Scenario B: median within the high-n_PA (reliable PA) subset
    valid = df.dropna(subset=["PA"]).copy()
    n_cutoff = valid["n_PA"].median()
    high_npa = valid[valid["n_PA"] >= n_cutoff]
    median_high_npa = high_npa["PR_AUC_mean"].median()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["PR_AUC_mean"], bins=25, color="#5f9ea0", edgecolor="white", alpha=0.85)

    ax.axvline(median_all, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Scenario A median (all 138): {median_all:.3f}")
    ax.axvline(median_high_npa, color="#2a78d6", linestyle="--", linewidth=1.5,
               label=f"Scenario B median (high n_PA, n={len(high_npa)}): {median_high_npa:.3f}")

    ax.set_xlabel("Baseline PR-AUC (best config per label)")
    ax.set_ylabel("Number of labels")
    ax.set_title("Distribution of per-label PR-AUC with proposed confidence thresholds")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Scenario A median (all 138 labels): {median_all:.4f}")
    print(f"Scenario B median (n_PA >= {n_cutoff:.1f}, n={len(high_npa)}): {median_high_npa:.4f}")


if __name__ == "__main__":
    main()
